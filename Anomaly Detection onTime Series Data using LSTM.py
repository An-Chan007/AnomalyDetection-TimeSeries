#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout, RepeatVector, TimeDistributed


# In[4]:


df = pd.read_csv('GOOG.csv')


# In[5]:


df.head()


# In[6]:


df = df[['Date', 'Close']]


# In[7]:


df.info()


# In[8]:


df['Date'].min(), df['Date'].max()


# In[9]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close price'))
fig.update_layout(showlegend=True, title='Apple Inc. Stock Price 2004-2020')
fig.show()


# # Preprocessing

# In[10]:


train = df.loc[df['Date']<= '2017-12-24']
test = df.loc[df['Date']>'2017-12-24']
train.shape, test.shape


# In[11]:


scaler = StandardScaler()
scaler = scaler.fit(np.array(train['Close']).reshape(-1,1))

train['Close'] = scaler.transform(np.array(train['Close']).reshape(-1,1))
test['Close'] = scaler.transform(np.array(test['Close']).reshape(-1,1))


# In[12]:


plt.plot(train['Close'], label = 'scaled')
plt.legend()
plt.show()


# In[13]:


TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    X_out, y_out = [], []
    for i in range(len(X)-time_steps):
        X_out.append(X.iloc[i:(i+time_steps)].values)
        y_out.append(y.iloc[i+time_steps])
        
    return np.array(X_out), np.array(y_out)

X_train, y_train = create_sequences(train[['Close']], train ['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])
print("Training input shape: ", X_train.shape)
print("Testing input shape: ", X_test.shape)


# In[14]:


np.random.seed(21)
tf.random.set_seed(21)


# # Model

# In[16]:


model = Sequential() 
model.add(LSTM(128, activation = 'tanh', input_shape =(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, activation = 'tanh', return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()


# In[18]:


history = model.fit(X_train,
                   y_train,
                   epochs=100,
                   batch_size=32,
                   validation_split=0.1,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                   shuffle=False)


# In[20]:


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend();


# In[21]:


X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples');

threshold = np.max(train_mae_loss)

print('Reconstruction error threshold:', threshold)


# In[22]:


X_test_pred = model.predict(X_test, verbose=1)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of Samples')


# In[24]:


anomaly_df = pd.DataFrame(test[TIME_STEPS:])
anomaly_df['loss'] = test_mae_loss
anomaly_df['threshold'] =threshold
anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']


# In[25]:


anomaly_df.head()


# In[29]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['loss'], name= 'Test loss'))
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['threshold'], name= 'Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs Threshold')
fig.show()


# In[30]:


anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
anomalies.head()


# In[ ]:





# In[ ]:




