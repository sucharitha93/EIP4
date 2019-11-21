>Number of parameters : ~8k

>Number of epocs : 20

>Accuracy Score : 99.46

*Approach*

Initially analyzed all the given models and understood the major contributors to the parameter count within the models.
Tried to reduce the size of parameters by reducing the kernel size in the initial convolution layers in the chosen model from the code 8.
Also to filter off the least percentage of irrelevant values, the drop off rate was raised from 0.1 to 0.15.
The learning rate was initalized with 0.002 within the scheduler.
These minute changes could result in a model with the expected accuracy within the expected time.



*Model Summary*
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_25 (Conv2D)           (None, 26, 26, 8)         72        
_________________________________________________________________
batch_normalization_22 (Batc (None, 26, 26, 8)         32        
_________________________________________________________________
dropout_22 (Dropout)         (None, 26, 26, 8)         0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 24, 24, 16)        1152      
_________________________________________________________________
batch_normalization_23 (Batc (None, 24, 24, 16)        64        
_________________________________________________________________
dropout_23 (Dropout)         (None, 24, 24, 16)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 24, 24, 8)         128       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 12, 12, 8)         0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 10, 10, 8)         576       
_________________________________________________________________
batch_normalization_24 (Batc (None, 10, 10, 8)         32        
_________________________________________________________________
dropout_24 (Dropout)         (None, 10, 10, 8)         0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 8, 8, 16)          1152      
_________________________________________________________________
batch_normalization_25 (Batc (None, 8, 8, 16)          64        
_________________________________________________________________
dropout_25 (Dropout)         (None, 8, 8, 16)          0         
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 6, 6, 16)          2304      
_________________________________________________________________
batch_normalization_26 (Batc (None, 6, 6, 16)          64        
_________________________________________________________________
dropout_26 (Dropout)         (None, 6, 6, 16)          0         
_________________________________________________________________
conv2d_31 (Conv2D)           (None, 4, 4, 8)           1152      
_________________________________________________________________
batch_normalization_27 (Batc (None, 4, 4, 8)           32        
_________________________________________________________________
dropout_27 (Dropout)         (None, 4, 4, 8)           0         
_________________________________________________________________
conv2d_32 (Conv2D)           (None, 1, 1, 10)          1280      
_________________________________________________________________
batch_normalization_28 (Batc (None, 1, 1, 10)          40        
_________________________________________________________________
dropout_28 (Dropout)         (None, 1, 1, 10)          0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0         
=================================================================
Total params: 8,144
Trainable params: 7,980
Non-trainable params: 164
```

*Epoc Output*

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.002.
60000/60000 [==============================] - 13s 221us/step - loss: 0.1678 - acc: 0.9174 - val_loss: 0.0272 - val_acc: 0.9922
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0015163002.
60000/60000 [==============================] - 6s 104us/step - loss: 0.1641 - acc: 0.9184 - val_loss: 0.0291 - val_acc: 0.9925
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0012210012.
60000/60000 [==============================] - 6s 101us/step - loss: 0.1633 - acc: 0.9192 - val_loss: 0.0251 - val_acc: 0.9930
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0010219724.
60000/60000 [==============================] - 6s 102us/step - loss: 0.1626 - acc: 0.9186 - val_loss: 0.0251 - val_acc: 0.9932
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0008787346.
60000/60000 [==============================] - 6s 103us/step - loss: 0.1595 - acc: 0.9199 - val_loss: 0.0267 - val_acc: 0.9933
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0007707129.
60000/60000 [==============================] - 6s 103us/step - loss: 0.1601 - acc: 0.9180 - val_loss: 0.0238 - val_acc: 0.9939
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0006863418.
60000/60000 [==============================] - 6s 105us/step - loss: 0.1584 - acc: 0.9192 - val_loss: 0.0244 - val_acc: 0.9937
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0006186205.
60000/60000 [==============================] - 6s 104us/step - loss: 0.1568 - acc: 0.9202 - val_loss: 0.0248 - val_acc: 0.9937
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0005630631.
60000/60000 [==============================] - 6s 106us/step - loss: 0.1549 - acc: 0.9213 - val_loss: 0.0257 - val_acc: 0.9933
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0005166624.
60000/60000 [==============================] - 6s 104us/step - loss: 0.1544 - acc: 0.9222 - val_loss: 0.0252 - val_acc: 0.9936
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.000477327.
60000/60000 [==============================] - 6s 106us/step - loss: 0.1572 - acc: 0.9202 - val_loss: 0.0231 - val_acc: 0.9937
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0004435573.
60000/60000 [==============================] - 6s 102us/step - loss: 0.1558 - acc: 0.9208 - val_loss: 0.0251 - val_acc: 0.9930
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0004142502.
60000/60000 [==============================] - 6s 102us/step - loss: 0.1550 - acc: 0.9214 - val_loss: 0.0220 - val_acc: 0.9941
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0003885759.
60000/60000 [==============================] - 6s 101us/step - loss: 0.1566 - acc: 0.9206 - val_loss: 0.0225 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0003658983.
60000/60000 [==============================] - 6s 100us/step - loss: 0.1539 - acc: 0.9219 - val_loss: 0.0222 - val_acc: 0.9940
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0003457217.
60000/60000 [==============================] - 6s 101us/step - loss: 0.1567 - acc: 0.9194 - val_loss: 0.0242 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000327654.
60000/60000 [==============================] - 6s 102us/step - loss: 0.1537 - acc: 0.9210 - val_loss: 0.0236 - val_acc: 0.9940
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.000311381.
60000/60000 [==============================] - 6s 103us/step - loss: 0.1561 - acc: 0.9208 - val_loss: 0.0228 - val_acc: 0.9943
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0002966479.
60000/60000 [==============================] - 6s 103us/step - loss: 0.1518 - acc: 0.9211 - val_loss: 0.0235 - val_acc: 0.9942
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000283246.
60000/60000 [==============================] - 6s 101us/step - loss: 0.1522 - acc: 0.9219 - val_loss: 0.0220 - val_acc: 0.9946
```

*Evaluation Score*
```
In [43]:
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.022015421112952755, 0.9946]
```
