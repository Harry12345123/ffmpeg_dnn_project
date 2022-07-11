#include <sys/time.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/dnn.hpp>
#include "opencv_queue.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavutil/time.h>
#include <libswresample/swresample.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/fifo.h>
#include <libavutil/imgutils.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

#define WIDTH 640
#define HEIGHT 480
#define CAMERA_DEV "/dev/video0"
#define CAMERA_FMT AV_PIX_FMT_YUYV422
#define ENCODE_FMT AV_PIX_FMT_YUV420P

using namespace dnn;

static pthread_mutex_t workmutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t av_fifo_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t opencv_dnn_mutex = PTHREAD_MUTEX_INITIALIZER;

OPENCV_QUEUE *opencv_queue = NULL;
OPENCV_QUEUE *handle_opencv_queue = NULL;

AVFifoBuffer *m_videoFifo = NULL;

std::vector<std::string> classes;

const std::string model_file = "./bvlc_googlenet.caffemodel";
const std::string config_file = "./opencv_extra/testdata/dnn/bvlc_googlenet.prototxt";
const std::string classes_file = "./classification_classes_ILSVRC2012.txt";

const std::string yolov3_model_file = "./coco.names";
const std::string yolov3_config_file = "./yolov3-tiny.cfg";
const std::string yolov3_weights = "./yolov3-tiny.weights";

CascadeClassifier cascade, nestedCascade;

float confThreshold = 0.5; //置信度阈值
float nmsThreshold = 0.4;  //非最大抑制阈值

void showDshowDeviceOption(char *devName)
{
    AVFormatContext *pFormatCtx = avformat_alloc_context();
    AVDictionary *options = NULL;
    av_dict_set(&options, "list_options", "true", 0);
    AVInputFormat *iformat = av_find_input_format("dshow");
    avformat_open_input(&pFormatCtx, devName, iformat, &options);
    avformat_free_context(pFormatCtx);
}

void *read_camera_thread(void *args)
{
    pthread_detach(pthread_self());
    avdevice_register_all();
    AVInputFormat *in_fmt = av_find_input_format("video4linux2");
    if (in_fmt == NULL)
    {
        printf("can't find_input_format\n");
        // return;
    }

    // 设置摄像头的分辨率
    AVDictionary *option = NULL;
    char video_size[10];
    sprintf(video_size, "%dx%d", WIDTH, HEIGHT);
    av_dict_set(&option, "video_size", video_size, 0);

    AVFormatContext *fmt_ctx = NULL;
    if (avformat_open_input(&fmt_ctx, CAMERA_DEV, in_fmt, &option) < 0)
    {
        printf("can't open_input_file\n");
        // return;
    }
    else
    {
        printf("Success Open Camera\n");
    }
    // printf device info
    av_dump_format(fmt_ctx, 0, CAMERA_DEV, 0);

    struct SwsContext *sws_ctx;
    // 图像格式转换：CAMERA_FMT --> ENCODE_FMT
    sws_ctx = sws_getContext(WIDTH, HEIGHT, CAMERA_FMT,
                             WIDTH, HEIGHT, ENCODE_FMT, 0, NULL, NULL, NULL);

    uint8_t *yuy2buf[4];
    int yuy2_linesize[4];
    int yuy2_size = av_image_alloc(yuy2buf, yuy2_linesize, WIDTH, HEIGHT, CAMERA_FMT, 1);

    uint8_t *yuv420pbuf[4];
    int yuv420p_linesize[4];
    int yuv420p_size = av_image_alloc(yuv420pbuf, yuv420p_linesize, WIDTH, HEIGHT, ENCODE_FMT, 1);

    // 初始化packet，存放编码数据
    AVPacket *camera_packet = av_packet_alloc();

    // 初始化frame，存放原始数据
    int y_size = WIDTH * HEIGHT;
    int frame_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, WIDTH, HEIGHT, 1);

    for (;;)
    {
        static int i = 0;
        // 摄像头获取图像数据
        av_read_frame(fmt_ctx, camera_packet);
        memcpy(yuy2buf[0], camera_packet->data, camera_packet->size);
        // 图像格式转化
        sws_scale(sws_ctx, (const uint8_t **)yuy2buf, yuy2_linesize,
                  0, HEIGHT, yuv420pbuf, yuv420p_linesize);
        av_packet_unref(camera_packet);

        if (av_fifo_space(m_videoFifo) >= frame_size)
        {
            pthread_mutex_lock(&workmutex);
            av_fifo_generic_write(m_videoFifo, yuv420pbuf[0], y_size, NULL);
            av_fifo_generic_write(m_videoFifo, yuv420pbuf[1], y_size / 4, NULL);
            av_fifo_generic_write(m_videoFifo, yuv420pbuf[2], y_size / 4, NULL);
            pthread_mutex_unlock(&workmutex);
        }

        usleep(200 * 10);
    }

    av_packet_free(&camera_packet);
    avformat_close_input(&fmt_ctx);
    sws_freeContext(sws_ctx);
    av_freep(yuy2buf);
    av_freep(yuv420pbuf);
}

void detectAndDraw(Mat &srcFrame, CascadeClassifier &cascade)
{
    Mat grayFrame;
    cvtColor(srcFrame, grayFrame, COLOR_BGR2GRAY);
    equalizeHist(grayFrame, grayFrame);

    cv::Size original_size;
    vector<Rect> rect;
    cascade.detectMultiScale(grayFrame, rect, 1.1, 10); // 分类器对象调用

    for (size_t i = 0; i < rect.size(); i++)
    {
        rectangle(srcFrame, rect[i].tl(), rect[i].br(), Scalar(255, 0, 255), 3);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
    //绘制边界框
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label; //边框上的类别标签与置信度
    }
    //绘制边界框上的标签
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

vector<cv::String> getOutputNames(const cv::dnn::Net &net)
{
    static vector<cv::String> names;
    if (names.empty())
    {
        //取得输出层指标
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<cv::String> layersNames = net.getLayerNames();
        //取得输出层名字
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); i++)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

void postprocess(cv::Mat &frame, const vector<cv::Mat> &outs)
{
    vector<int> classIds;      //储存识别类的索引
    vector<float> confidences; //储存置信度
    vector<cv::Rect> boxes;    //储存边框

    for (size_t i = 0; i < outs.size(); i++)
    {
        //从网络输出中扫描所有边界框
        //保留高置信度选框
        //目标数据data:x,y,w,h为百分比，x,y为目标中心点坐标
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence; //置信度
            //取得最大分数值与索引
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    //低置信度
    vector<int> indices; //保存没有重叠边框的索引
    //该函数用于抑制重叠边框
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}

void opencv_dnn_process(Mat &srcFrame)
{
    Net net = readNetFromDarknet(yolov3_config_file, yolov3_weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);

    bool swap_rb = true;
    int train_width = 320;
    int train_height = 320;
    Scalar mean = Scalar(104.0, 117.0, 123.0);

    Mat blob;
    float scale = 1.0;

    blobFromImage(srcFrame, blob, 1 / 255.0, Size(train_width, train_height), mean, swap_rb, false);
    // blobFromImage(srcFrame, blob, 1 / 255.0, Size(416, 416));
    //   设置网络
    net.setInput(blob);
    // 向前预测
    vector<cv::Mat> outs; //储存识别结果
    net.forward(outs, getOutputNames(net));
    postprocess(srcFrame, outs);

    vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = cv::format("Infercence time for a frame:%.2f ms", t);
    cv::putText(srcFrame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
}

void *process_opencv_thread(void *args)
{
    pthread_detach(pthread_self());
    vector<Rect> rect;

    AVFrame *dst, *yuv422_final_frame;
    // cv::Size frameSize = frame.size();

    while (1)
    {
        if (opencv_queue->getMatQueueSize() > 0)
        {
            Mat mat = opencv_queue->getMatQueue();
            cv::Size frameSize = mat.size();
            dst = av_frame_alloc();

            avpicture_fill((AVPicture *)dst, mat.data, AV_PIX_FMT_BGR24, WIDTH, HEIGHT);
            dst->width = frameSize.width;
            dst->height = frameSize.height;

            yuv422_final_frame = av_frame_alloc();
            int yuv422_num_bytes = avpicture_get_size(AV_PIX_FMT_YUV420P, dst->width, dst->height);
            uint8_t *final_frame2_buffer = (uint8_t *)av_malloc(yuv422_num_bytes * sizeof(uint8_t));

            yuv422_final_frame->width = frameSize.width;
            yuv422_final_frame->height = frameSize.height;
            avpicture_fill((AVPicture *)yuv422_final_frame, (uchar *)dst->data, AV_PIX_FMT_YUV420P, dst->width, dst->height);

            SwsContext *final_sws_ctx = sws_getContext(dst->width, dst->height,
                                                       AV_PIX_FMT_BGR24, dst->width, dst->height,
                                                       AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, 0, 0, 0);

            sws_scale(final_sws_ctx, dst->data,
                      dst->linesize,
                      0, dst->height,
                      yuv422_final_frame->data,
                      yuv422_final_frame->linesize);

            // av_free(final_frame2_buffer);
            av_frame_free(&dst);
            av_frame_free(&yuv422_final_frame);
        }
    }
}

void *process_ffmpeg_thread(void *args)
{
    pthread_detach(pthread_self());
    int y_size = WIDTH * HEIGHT;
    int frame_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, WIDTH, HEIGHT, 1);
    uint8_t *out_buffer_yuv420 = (uint8_t *)av_malloc(frame_size);

    // 初始化编码器
    AVCodec *cod = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (cod == NULL)
    {
        printf("failed to find encoder\n");
        // return;
    }

    AVCodecContext *cod_ctx = avcodec_alloc_context3(cod);
    cod_ctx->pix_fmt = ENCODE_FMT;
    cod_ctx->width = WIDTH;
    cod_ctx->height = HEIGHT;
    cod_ctx->time_base.num = 1;
    cod_ctx->time_base.den = 30;
    cod_ctx->bit_rate = 400000;
    cod_ctx->qmin = 10;
    cod_ctx->qmax = 51;
    cod_ctx->max_b_frames = 0;
    cod_ctx->thread_count = 4;
    cod_ctx->gop_size = 15;

    if (avcodec_open2(cod_ctx, cod, NULL) < 0)
    {
        printf("failed to open encoder\n");
        // return;
    }

    AVPacket *packet = av_packet_alloc();

    FILE *fp = fopen("output_camera.h264", "w+");

    while (1)
    {
        if (av_fifo_size(m_videoFifo) >= frame_size)
        {
            static int i = 0;
            int video_fifo_size = av_fifo_size(m_videoFifo);
            pthread_mutex_lock(&workmutex);
            av_fifo_generic_read(m_videoFifo, out_buffer_yuv420, frame_size, NULL);
            pthread_mutex_unlock(&workmutex);

            AVFrame *frame = av_frame_alloc();
            frame->format = AV_PIX_FMT_YUV420P;
            frame->width = 640;
            frame->height = 480;

            // 为AVFrame分配内存，调用此函数前必须先设置format;width/height(video);nb_samples/channel_layout(audio)
            // 如果AVFrame已经分配了内存，再次调用会造成内存泄漏和不可预知错误；参数二传0即可，表示根据目前cpu类型自动选择对齐的字节数
            av_frame_get_buffer(frame, 0);
            // 让Frame可写
            av_frame_make_writable(frame);

            frame->data[0] = out_buffer_yuv420;
            frame->data[1] = frame->data[0] + y_size;
            frame->data[2] = frame->data[1] + y_size / 4;
            frame->pts = i++;

            int send_result = avcodec_send_frame(cod_ctx, frame);
            if (send_result < 0)
            {
                printf("send failed:%d.\n", send_result);
            }

            while (send_result >= 0)
            {
                // 编码器对图像数据进行编码
                int receive_result = avcodec_receive_packet(cod_ctx, packet);
                if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF)
                    break;
                packet->pts = i++;
                fwrite(packet->data, 1, packet->size, fp);
                printf("get %d frames\n", i);
                av_packet_unref(packet);
            }
            av_frame_free(&frame);
        }
    }
}

void *process_avframe_dnn_thread(void *args)
{
    pthread_detach(pthread_self());

    int y_size = WIDTH * HEIGHT;
    int frame_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, WIDTH, HEIGHT, 1);
    uint8_t *out_buffer_yuv420 = (uint8_t *)av_malloc(frame_size);

    while (1)
    {
        if (av_fifo_size(m_videoFifo) >= frame_size)
        {
            static int i = 0;
            int video_fifo_size = av_fifo_size(m_videoFifo);
            pthread_mutex_lock(&workmutex);
            av_fifo_generic_read(m_videoFifo, out_buffer_yuv420, frame_size, NULL);
            pthread_mutex_unlock(&workmutex);

            int width = WIDTH, height = HEIGHT;

            cv::Mat tmp_img = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
            //cv::Mat tmp_img = cv::Mat::zeros(height, width, CV_8UC1);
            Mat mainRgbImage;
            memcpy(tmp_img.data, out_buffer_yuv420, frame_size);
            cv::cvtColor(tmp_img, mainRgbImage, COLOR_YUV2BGR_I420);
            opencv_dnn_process(mainRgbImage);
            opencv_queue->putMatQueue(mainRgbImage);
            tmp_img.release();
        }
    }
}

void *process_mat_dnn_thread(void *args)
{
    pthread_detach(pthread_self());

    while (1)
    {
        if (handle_opencv_queue->getMatQueueSize() > 0)
        {
            Mat tmpRgbImage = handle_opencv_queue->getMatQueue();
            opencv_dnn_process(tmpRgbImage);
            opencv_queue->putMatQueue(tmpRgbImage);
            // usleep(200);
        }
    }
}

void *show_opencv_thread(void *args)
{
    pthread_detach(pthread_self());
    vector<Rect> rect;
    while (1)
    {
        if (opencv_queue->getMatQueueSize() > 0)
        {
            Mat tmp = opencv_queue->getMatQueue();
            cv::imshow("bgr_show", tmp);
            waitKey(1);
            tmp.release();
        }
    }
}

int main(int argc, char *argv[])
{

    std::ifstream ifs(yolov3_model_file.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "File " + yolov3_model_file + " not found";
        return -1;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    int ret;
    m_videoFifo = av_fifo_alloc(15 * av_image_get_buffer_size(AV_PIX_FMT_YUV420P, 640, 480, 1));

    opencv_queue = new OPENCV_QUEUE();
    handle_opencv_queue = new OPENCV_QUEUE();

    pthread_t pid;
    ret = pthread_create(&pid, NULL, read_camera_thread, NULL);
    if (ret != 0)
    {
    }

    ret = pthread_create(&pid, NULL, process_avframe_dnn_thread, NULL);
    if (ret != 0)
    {
    }
 
    ret = pthread_create(&pid, NULL, show_opencv_thread, NULL);
    if (ret != 0)
    {
    }

    while (1)
    {
        sleep(1000);
    }

    return 0;
}
