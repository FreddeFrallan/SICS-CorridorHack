start_animation() (
    cmd="cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display ; env DISPLAY=:1 PATH=/home/fredrik/anaconda3/bin:$PATH ipython animation.py localhost 5001"
     sudo su -l fredrik -c bash -c "$cmd"
)

old_start_img2txt() (
    cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display/MultiStreamModel
    CUDA_VISIBLE_DEVICES=0  PYTHONPATH=..:../../Img2Txt python img2Txt.py
)

start_img2txt() (
    cd ~larsr/SICS-CorridorHack/Demo/display/annotate

    url1='https://www.youtube.com/watch?v=HE9nLWFZ6ac'
    python send_img2txt.py --frame_nr 1 --url $url1 &

    url3='https://www.youtube.com/watch?v=MjyDHXOUdGc'
    python send_img2txt.py --frame_nr 3 --url $url3
)
    


start_facedetect() (
    cd ~fredrik/Desktop/SICS-CorridorHack/Demo/display/MultiStreamModel/ImageDetection
    CUDA_VISIBLE_DEVICES=0  PYTHONPATH=../..:../../../Img2Txt python PredictionModel.py
)

start_camera() (
    cd ~larsr/SICS-CorridorHack/Demo/display/face_recognition
    #taskset --cpu-list 6-16 env CUDA_VISIBLE_DEVICES=1 ipython face_recog.py
    python face_recog.py --add_faces img  --remote_display localhost:5001
)


start_YOLO1() (
    cd ~larsr/SICS-CorridorHack/Demo/display/YOLO
    ABCNEWS='https://www.youtube.com/watch?v=rQSwh3bgs5k'
    taskset --cpu-list 5 env CUDA_VISIBLE_DEVICES=1 ipython send_yolo.py localhost 5001 6  "$ABCNEWS"
)

start_YOLO2() (
    cd ~larsr/SICS-CorridorHack/Demo/display/YOLO
    TRAFFIC='https://www.youtube.com/watch?v=94OODNhoXN4'
    taskset --cpu-list 4 env CUDA_VISIBLE_DEVICES=1 ipython send_yolo.py localhost 5001 5  "$TRAFFIC"
)



$@
