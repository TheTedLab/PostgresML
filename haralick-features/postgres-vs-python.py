import numpy as np
import matplotlib.pyplot as plt

data = [
    [ 15085, 9166, 47852, 12522, 29907, 1525],
    [   980, 3084, 20252,  1292,  9170,  384]
]

bar_data = [
    [],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
]

for i in range(len(data[0])):
    bar_data[0].append(data[0][i] / data[1][i])

columns = (
    'Загрузка набора данных', 'Создание цифровых образов', 'Внесение шума',
    'Создание модели', 'Обучение модели', 'Тестирование модели'
)
rows = ['Postgres + PL/Python', 'Python']

colors = ['green', 'gray']
index = np.arange(len(columns)) + 0.166
bar_width = 0.2
y_offset = np.zeros(len(columns))

values = np.arange(0, 17.5, 0.5)
plt.figure(figsize=(19.2, 10.8), dpi=100)
rects_0 = plt.bar(index - 0.12, bar_data[1], bar_width, bottom=y_offset, color=colors[1], label=rows[1])
plt.bar_label(rects_0, fmt='%1.1f', fontsize=16)
rects_1 = plt.bar(index + 0.12, bar_data[0], bar_width, bottom=y_offset, color=colors[0], label=rows[0])
plt.bar_label(rects_1, fmt='%1.1f', fontsize=16)
plt.legend(loc='upper center', ncols=2, fontsize=16)

plt.ylabel(f"Время исполнения, кол-во раз", fontdict={'fontsize': 20})
plt.yticks(values, fontsize=16)
plt.xticks([0.166, 1.166, 2.166, 3.166, 4.166, 5.166], columns, fontsize=14)
plt.title('Сравнение ML задач в Python и Postgres + PL/Python', fontdict={'fontsize': 20})
plt.tight_layout()
plt.savefig('graphs/postgres-vs-python-graph.png', bbox_inches='tight')
plt.show()

plt.clf()

values = np.arange(0, 54000, 2000)
plt.figure(figsize=(19.2, 10.8), dpi=100)
rects_2 = plt.bar(index - 0.12, data[0], bar_width, bottom=y_offset, color=colors[1], label=rows[1])
plt.bar_label(rects_2, fmt='%d', fontsize=16)
rects_3 = plt.bar(index + 0.12, data[1], bar_width, bottom=y_offset, color=colors[0], label=rows[0])
plt.bar_label(rects_3, fmt='%d', fontsize=16)
plt.legend(loc='upper center', ncols=2, fontsize=16)

plt.ylabel(f"Время исполнения, миллисекунды", fontdict={'fontsize': 20})
plt.yticks(values, fontsize=16)
plt.xticks([0.166, 1.166, 2.166, 3.166, 4.166, 5.166], columns, fontsize=14)
plt.title('Сравнение ML задач в Python и Postgres + PL/Python', fontdict={'fontsize': 20})
plt.tight_layout()
plt.savefig('graphs/postgres-vs-python-graph-ms.png', bbox_inches='tight')
plt.show()

