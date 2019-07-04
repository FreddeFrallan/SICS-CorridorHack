start_animation() (
    DISPLAY="${1:-:0}"
    cmd="cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display ; env DISPLAY=$DISPLAY PATH=/home/fredrik/anaconda3/bin:$PATH ipython animation.py localhost 5001"
    echo "$cmd"
     sudo su -l fredrik -c bash -c "$cmd"
)

old_start_img2txt() (
    cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display/MultiStreamModel
    CUDA_VISIBLE_DEVICES=0  PYTHONPATH=..:../../Img2Txt python img2Txt.py
)

start_img2txt1() (
    cd ~larsr/SICS-CorridorHack/Demo/display/annotate
    python send_img2txt.py --frame_nr 1 --url 'https://www.youtube.com/watch?v=ZGn6R_uvE3E'
)

start_img2txt2() (
    cd ~larsr/SICS-CorridorHack/Demo/display/annotate
    python send_img2txt.py --frame_nr 3 --url 'https://www.youtube.com/watch?v=XePABzpXUi8'
)

start_posedemo() (
    cd ~larsr/SICS-CorridorHack/Demo/display/pose
    python posenetdemo.py
)

start_facedetect() (
    cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display/MultiStreamModel/ImageDetection
    CUDA_VISIBLE_DEVICES=0  PYTHONPATH=../..:../../../Img2Txt python PredictionModel.py
)

start_camera() (
    cd ~larsr/SICS-CorridorHack/Demo/display/face_recognition
    taskset --cpu-list 6-16 env CUDA_VISIBLE_DEVICES=1 \
    python face_recog.py --add_faces img  --remote_display localhost:5001
)


start_YOLO1() (
    cd ~larsr/SICS-CorridorHack/Demo/display/YOLO
    ABCNEWS='https://www.youtube.com/watch?v=E6YND-6Gtv0'
    SKEDSMO="https://www.youtube.com/watch?v=tbLXWVhu8-Q"
    taskset --cpu-list 5 env CUDA_VISIBLE_DEVICES=1 ipython send_yolo.py localhost 5001 6  "$SKEDSMO"
)

start_YOLO2() (
    cd ~larsr/SICS-CorridorHack/Demo/display/YOLO
    TRAFFIC='https://www.youtube.com/watch?v=94OODNhoXN4'
    taskset --cpu-list 4 env CUDA_VISIBLE_DEVICES=1 ipython send_yolo.py localhost 5001 5  "$TRAFFIC"
)

start_runner() (
     cd ~larsr/SICS-CorridorHack/Demo/display/runner
     taskset --cpu-list 4 env CUDA_VISIBLE_DEVICES=1 python runner.py
)


$@
