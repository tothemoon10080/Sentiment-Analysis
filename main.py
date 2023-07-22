from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# 加载IMDb电影评论数据集
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

# 文本预处理和序列填充
maxlen = 200  # 假设评论最大长度为200个单词
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# 构建情感分析模型
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=maxlen))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.2)

# 在测试集上评估模型
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {accuracy}')

# 进行预测
new_texts = ["This is a really good movie!", "This movie is really bad."]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(new_texts)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_sequences_padded = pad_sequences(new_sequences, maxlen=maxlen)
predictions = model.predict(new_sequences_padded)

for i, text in enumerate(new_texts):
    sentiment = "正面" if predictions[i] > 0.5 else "负面"
    print(f"\"{text}\" 的情感倾向为：{sentiment}")