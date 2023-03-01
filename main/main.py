from tkinter import *
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Создание окна
root = Tk()
root.title("Категоризация текста")

# Исходные данные
categories = ['food', 'technology', 'politics', 'travel']
train_data = ['This is a article about food','pizza, sushi and burgers are very tasty' ,'This is a technology article.', 'This is a political article.', 'This is a travel article.']
train_labels = [0, 0, 1, 2, 3]

# Векторизация текста
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)

# Обучение модели
model = LinearSVC()
model.fit(train_vectors, train_labels)

# Функция для предсказания категории текста и вывода результата
def predict_category():
    new_text = text_entry.get()
    vectorized_text = vectorizer.transform([new_text])
    predicted_category = model.predict(vectorized_text)
    result_label.configure(text=categories[predicted_category[0]])

# Функция для оценки качества модели на тестовых данных и вывода результата
def evaluate_model():
    test_data = ['This is another food article.', 'This is another technology article.', 'This is another political article.', 'This is another travel article.']
    test_labels = [0, 1, 2, 3]
    test_vectors = vectorizer.transform(test_data)
    predicted_labels = model.predict(test_vectors)
    report = classification_report(test_labels, predicted_labels, target_names=categories)
    result_label.configure(text=report)

# Создание элементов интерфейса
text_label = ttk.Label(root, text="Введите текст:")
text_entry = ttk.Entry(root)
predict_button = ttk.Button(root, text="Предсказать категорию", command=predict_category)
evaluate_button = ttk.Button(root, text="Оценить качество модели", command=evaluate_model)
result_label = ttk.Label(root, text="")

# Размещение элементов интерфейса
text_label.pack()
text_entry.pack()
predict_button.pack()
evaluate_button.pack()
result_label.pack()

# Запуск главного цикла обработки событий
root.mainloop()
