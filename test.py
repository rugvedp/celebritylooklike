
from keras_vggface.vggface import VGGFace
vggface = VGGFace(model='resnet50')

#print(vggface.summary())
vggface.save('vggmodel.h5')