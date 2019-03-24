# Img2TXT demo

Uses the [Show & Tell model](https://github.com/tensorflow/models/tree/master/research/im2txt)

The pre-trained weights for the model can be [downloaded here](https://drive.google.com/file/d/15YdxTRH_zOU2yZPTNwcGtrubB8CWszTF/view?usp=sharing).

1. Download the pre-trained weights and unzip them in the <b>"SICS-COrridorHack/Demo/Img2Txt/Data"</b> folder.
2. Start a [SimpleCameraProducer](https://github.com/FreddeFrallan/SICS-CorridorHack/blob/master/Demo/CameraStream/SimpleCameraProducer.py) and take note of what IP & Port your streaming to.
3. Start a [Img2Txt Producer](https://github.com/FreddeFrallan/SICS-CorridorHack/blob/master/Demo/Img2Txt/ProducerNode.py) and specify the previous IP & Port as your input. Also take note of what IP & Port this node will be streaming its output to.
4. Start one or several [Img2Txt Consumer](https://github.com/FreddeFrallan/SICS-CorridorHack/blob/master/Demo/Img2Txt/ConsumerNode.py) that listens to the <b>Img2Txt Producer</b> that you just configured.


The current workflow with having to specify adresses and ports is soon to be replaced with an [automatic version](https://github.com/FreddeFrallan/SICS-CorridorHack/blob/master/Demo/hello.py)
