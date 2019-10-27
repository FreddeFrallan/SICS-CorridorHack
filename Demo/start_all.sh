#
# The account fredrik is set to autologin.
#
# In fredrik's crontab, there is this line:
# @reboot screen -S x11vnc -dm  bash -c "while true; do x11vnc -forever -display :0; done"
# which exports the display :0 over VNC port 127.0.0.1:5900
#
# To give larsr the ability to open windows at fredrik's screen, do
# sudo -u fredrik env DISPLAY=:0 xhost +si:localuser:larsr

start_animation() (
    export DISPLAY="${1:-:0}"
    cd ~/SICS-CorridorHack/Demo/display
    python animation.py 5001
)

start_animation_old() (
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
    ABCNEWS='https://www.youtube.com/watch?v=nu3iXDR7iXo'

    python send_img2txt.py --frame_nr 1 --url  "$ABCNEWS"
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
    BEIJING="https://www.youtube.com/watch?v=v0rY4x87xfs"
    disp=6
    taskset --cpu-list 5 env CUDA_VISIBLE_DEVICES=1 \
    ipython send_yolo.py localhost 5001 $disp  "$BEIJING"

)

start_YOLO2() (
    cd ~larsr/SICS-CorridorHack/Demo/display/YOLO
    TRAFFIC='https://www.youtube.com/watch?v=94OODNhoXN4'
    RAPPORT='https://youtu.be/ZB7h4YCSFXw'
    disp=3
    taskset --cpu-list 4 env CUDA_VISIBLE_DEVICES=1 \
    ipython send_yolo.py localhost 5001 $disp  "$RAPPORT"
)

start_runner() (
     cd ~larsr/SICS-CorridorHack/Demo/display/runner
     taskset --cpu-list 4 env CUDA_VISIBLE_DEVICES=1 python runner.py
)


if [ "$@" ] ; then
    
    start_$@

else
    grep ^start_ $0 | cut -c 7- | cut -d\( -f1
fi
