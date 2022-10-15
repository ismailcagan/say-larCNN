# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:24:07 2021

@author: ismail
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")



train=pd.read_csv("train.csv")

test=pd.read_csv("test.csv")


print(train.shape)
print(train.info())
print(train.columns)
print(train.head())


print(test.shape)
print(test.info())
print(test.columns)
print(test.head())



y_train=train["label"]

x_train=train.drop(labels=["label"],axis=1)



plt.figure(figsize=(15,7))
g=sns.countplot(y_train,palette="icefire")
plt.title("number of digit classes")

y_train.value_counts()



img=x_train.iloc[0].to_numpy()
img=img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

img=x_train.iloc[1].to_numpy()
img=img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[1,0])
plt.axis("off")
plt.show()


img=x_train.iloc[2].to_numpy()
img=img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[2,0])
plt.axis("off")
plt.show()


img=x_train.iloc[3].to_numpy()
img=img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()





# normalizasion : 
    
    ## normalisazyon yapmaksak farklı renklerden oluşan hatalar meydana gelebilir
    ## normalisazyon yapmak garyskay yapmak 0 ile 1 arasında sayılara çevirmek
    
# not : keras 28*28 anlamaz , 28*28*1 anlar, keras bu formatta çalışıyor
    

x_train=x_train/255.0
test=test/255.0
print("x_train shape :",x_train.shape)
print("test shape :",test.shape)



x_train=x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

print("x_train shape :",x_train.shape)
print("test shape :",test.shape)



from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train,num_classes=10)   # 0 ve 1 formatına çevirir , kerasın anlaması için



# train ve test split
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=2)

# X_VALUDEŞIN :
# x_train model oluşacak , x_val :bu model test edilecek ,daha sonra test datasını test ediceğiz


print("x_train shape :",x_train.shape)
print("x_test shape :",x_val.shape)
print("y_train shape :",y_train.shape)
print("y_test shape :",y_val.shape)


# keras kütüphanesi ile kod yazma

# ilk modelimizi yaradtacağız,kullanılan model aşağıda gösterildi


# conv = >maxpool => dropout => conv = >maxpool = >dropout => fully


# droupaut , defterde yazılı

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam # optimasyon
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# cnn
"""     1 model yarattık """

model=Sequential() # içerisinde layerları barındıran yapı .model yapmak için kullanılıyor

# conv eklendi
model.add(Conv2D(filters=8,kernel_size=(5,5),padding="valid",activation="relu",input_shape=(28,28,1)))

# filters => 8 tane filtre kullandık
# kernel_size => filtre boyutu 5,5 lik kullandık
# padding => boyutu kaybetmemek için same padding kullandık
# activation => activation fonksiyonu relu kullandık

# maxpool eklendi
model.add(MaxPool2D(pool_size=(2,2)))

# droupaut yapıldı
model.add(Dropout(0.25)) # randomuk oranı 4te 1 kapat ,randomluk 4 te 1 tutarsa



# tukarıdakini tekrarlıcağız
model.add(Conv2D(filters=16,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# strides => convulationu gezdirirken ne kadar atlayacağız (2,2) anlayacağız
model.add(Dropout(0.25))



#fully connected

model.add(Flatten()) # tek sütun haline getirme
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

model.summary() # modeli gösteriyor


# neden softmax=> sigmoid in genişletilmiş hali ,sigmoidden 2 şeyi ayırt etmek için kullanılıyor ama softmax kedi,köpek ,fare ,aslan gibi bir çok şeyi ayırt eder


"""     2. define optimizer  (optimize edici)    öğretmek için kullanılıyor   """
# learning rate değiştirerek daha hızlı öğrenmemizi sağlıyor
# cost değerine hızlı gidemiyorsak learning rate artırıyor , çok hızlı gittiysek yavaşlatıyor

optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999)
# betalar : learning rate değişimini sağlıyor

#model.fit()

"""     3.compile model(derleme modeli) parçaları birleştireceğiz  """


model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"]) # ***çalış buraya


# batch and epoch

# elimizde 10 tane resim var biz batch_size=2 olarak belirledik. her seferinde forward ve backward propagation yaparken
# 2 resimle yapılır her seferde 2 resimle yapılır ve toplam 5 kez batch yapılması gerekir.

# burada 1 epochk ta 5 kez batch yapmış olacağız, 3 epochta 15 kez batch yapmış olacağız

epochs=10
batch_size=250


# Data Augmentation (veri büyütme) ==> yeni veri ekleme,çeşitliliği artırma

datagen=ImageDataGenerator(
    
    featurewise_center=False,                 
    samplewise_center=False,           
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.5,
    zoom_range=0.5,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=False,
    vertical_flip=False
    
    )

datagen.fit(x_train)   # uyguluyoruz


history=model.fit_generator(datagen.flow(x_train,y_train,batch_size),epochs = epochs,validation_data=(x_val,y_val),steps_per_epoch=x_train.shape[0]//batch_size)


# evaluate the model (modeli değerlendirme)


plt.plot(history.history["val_loss"],color="b",label="validation") # bu değerler içindeki değerler
plt.title("test loss")
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.legend()
plt.show()



import seaborn as sns

y_pred=model.predict(x_val)
y_pred_classes=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_val,axis=1)
confusion_mtx=confusion_matrix(y_true , y_pred_classes)

f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx,annot=True, linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)

plt.xlabel("predicted label")
plt.ylabel("true label")
plt.title("confusion matrix")
plt.show()

























































































































