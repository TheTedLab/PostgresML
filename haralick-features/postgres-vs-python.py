import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn")

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
    'Загрузка набора\nданных', 'Создание цифровых\nобразов', 'Внесение\nшума',
    'Создание\nмодели', 'Обучение\nмодели', 'Тестирование\nмодели'
)
rows = ['Postgres + PL/Python', 'Python']

colors = ['green', 'gray']
index = np.arange(0.0, len(columns) * 1.0, 1.0) + 0.166
bar_width = 0.2
y_offset = np.zeros(len(columns))

values = np.arange(0, 17.5, 0.5)
plt.figure(figsize=(19.2, 10.8), dpi=100)
rects_0 = plt.bar(index - 0.12, bar_data[1], bar_width, bottom=y_offset, color=colors[1], label=rows[1])
plt.bar_label(rects_0, fmt='%.1f', fontsize=20)
rects_1 = plt.bar(index + 0.12, bar_data[0], bar_width, bottom=y_offset, color=colors[0], label=rows[0])
plt.bar_label(rects_1, fmt='%.1f', fontsize=20)
plt.legend(loc='upper center', ncols=2, fontsize=20)

plt.ylabel(f"Уменьшение времени выполнения, кол-во раз", fontdict={'fontsize': 20})
plt.yticks(values, fontsize=20)
plt.xticks(index, columns, fontsize=20)
plt.title('Сравнение времени ML задач в Python и Postgres + PL/Python (больше лучше)', fontdict={'fontsize': 20})
plt.tight_layout()
plt.savefig('graphs/postgres-vs-python-graph.png', bbox_inches='tight')
plt.show()

plt.clf()

values = np.arange(0.0, 56.0, 2.0)
plt.figure(figsize=(19.2, 10.8), dpi=100)
rects_2 = plt.bar(index - 0.12, [x / 1000.0 for x in data[0]], bar_width, bottom=y_offset, color=colors[1], label=rows[1])
plt.bar_label(rects_2, fmt='%.1f', fontsize=20)
rects_3 = plt.bar(index + 0.12, [x / 1000.0 for x in data[1]], bar_width, bottom=y_offset, color=colors[0], label=rows[0])
plt.bar_label(rects_3, fmt='%.1f', fontsize=20)
plt.legend(loc='upper center', ncols=2, fontsize=20)

plt.ylabel(f"Время выполнения, секунды", fontdict={'fontsize': 20})
plt.yticks(values, fontsize=20)
plt.xticks(index, columns, fontsize=20)
plt.title('Сравнение времени ML задач в Python и Postgres + PL/Python (меньше лучше)', fontdict={'fontsize': 20})
plt.tight_layout()
plt.savefig('graphs/postgres-vs-python-graph-s.png', bbox_inches='tight')
plt.show()

values = np.arange(0, 17.5, 0.5)

frames = 120.0

interval_list = []
for i in range(0, 6):
    interval_list.append([bar_data[1][i] / frames, bar_data[0][i] / frames])

lst_ranges = []
for i in range(0, 6):
    lst_ranges.append([
        np.arange(0.0, bar_data[1][i] + interval_list[i][0], interval_list[i][0]),
        np.arange(0.0, bar_data[0][i] + interval_list[i][1], interval_list[i][1])
    ])

index_list = []
for i in range(0, 6):
    index_list.append(np.arange(i, i + 1.0) + 0.5)

y_dict = {
    '0': [], '1': [],
    '2': [], '3': [],
    '4': [], '5': [],
    '6': [], '7': [],
    '8': [], '9': [],
    '10': [], '11': []
}

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
def animate_one(i):
    print(i)
    plt.clf()

    if 0 <= i < 120:
        y_dict['0'], y_dict['1'] = lst_ranges[0][0][i], lst_ranges[0][1][i]
        y_dict['2'], y_dict['3'] = lst_ranges[1][0][i], lst_ranges[1][1][i]
        y_dict['4'], y_dict['5'] = lst_ranges[2][0][i], lst_ranges[2][1][i]
        y_dict['6'], y_dict['7'] = lst_ranges[3][0][i], lst_ranges[3][1][i]
        y_dict['8'], y_dict['9'] = lst_ranges[4][0][i], lst_ranges[4][1][i]
        y_dict['10'], y_dict['11'] = lst_ranges[5][0][i], lst_ranges[5][1][i]
    else:
        y_dict['0'], y_dict['1'] = lst_ranges[0][0][-1], lst_ranges[0][1][-1]
        y_dict['2'], y_dict['3'] = lst_ranges[1][0][-1], lst_ranges[1][1][-1]
        y_dict['4'], y_dict['5'] = lst_ranges[2][0][-1], lst_ranges[2][1][-1]
        y_dict['6'], y_dict['7'] = lst_ranges[3][0][-1], lst_ranges[3][1][-1]
        y_dict['8'], y_dict['9'] = lst_ranges[4][0][-1], lst_ranges[4][1][-1]
        y_dict['10'], y_dict['11'] = lst_ranges[5][0][-1], lst_ranges[5][1][-1]

    rect0 = plt.bar(index_list[0] - 0.12, y_dict['0'], bar_width, color='gray', label=rows[1])
    plt.bar_label(rect0, fmt='%1.1f', fontsize=20)
    rect1 = plt.bar(index_list[0] + 0.12, y_dict['1'], bar_width, color='green', label=rows[0])
    plt.bar_label(rect1, fmt='%1.1f', fontsize=20)
    rect2 = plt.bar(index_list[1] - 0.12, y_dict['2'], bar_width, color='gray')
    plt.bar_label(rect2, fmt='%1.1f', fontsize=20)
    rect3 = plt.bar(index_list[1] + 0.12, y_dict['3'], bar_width, color='green')
    plt.bar_label(rect3, fmt='%1.1f', fontsize=20)
    rect4 = plt.bar(index_list[2] - 0.12, y_dict['4'], bar_width, color='gray')
    plt.bar_label(rect4, fmt='%1.1f', fontsize=20)
    rect5 = plt.bar(index_list[2] + 0.12, y_dict['5'], bar_width, color='green')
    plt.bar_label(rect5, fmt='%1.1f', fontsize=20)
    rect6 = plt.bar(index_list[3] - 0.12, y_dict['6'], bar_width, color='gray')
    plt.bar_label(rect6, fmt='%1.1f', fontsize=20)
    rect7 = plt.bar(index_list[3] + 0.12, y_dict['7'], bar_width, color='green')
    plt.bar_label(rect7, fmt='%1.1f', fontsize=20)
    rect8 = plt.bar(index_list[4] - 0.12, y_dict['8'], bar_width, color='gray')
    plt.bar_label(rect8, fmt='%1.1f', fontsize=20)
    rect9 = plt.bar(index_list[4] + 0.12, y_dict['9'], bar_width, color='green')
    plt.bar_label(rect9, fmt='%1.1f', fontsize=20)
    rect10 = plt.bar(index_list[5] - 0.12, y_dict['10'], bar_width, color='gray')
    plt.bar_label(rect10, fmt='%1.1f', fontsize=20)
    rect11 = plt.bar(index_list[5] + 0.12, y_dict['11'], bar_width, color='green')
    plt.bar_label(rect11, fmt='%1.1f', fontsize=20)

    plt.legend(loc='upper center', ncols=2, fontsize=20)
    plt.ylabel(f"Уменьшение времени выполнения, кол-во раз", fontdict={'fontsize': 20})
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], columns, fontsize=20)
    plt.yticks(values, fontsize=20)
    plt.title('Сравнение времени ML задач в Python и Postgres + PL/Python (больше лучше)', fontdict={'fontsize': 20})
    plt.ylim([0, 18])
    plt.xlim([0.0, 6.0])
    plt.tight_layout()

ani = FuncAnimation(fig, animate_one, frames=240, interval=10)
ani.save('postgres-vs-python.mp4', writer='ffmpeg', fps=60)

values = np.arange(0.0, 56.0, 2.0)

frames = 120.0

interval_list = []
for i in range(0, 6):
    interval_list.append([data[0][i] / 1000.0 / frames, data[1][i] / 1000.0 / frames])

lst_ranges = []
for i in range(0, 6):
    lst_ranges.append([
        np.arange(0.0, data[0][i] / 1000.0 + interval_list[i][0], interval_list[i][0]),
        np.arange(0.0, data[1][i] / 1000.0 + interval_list[i][1], interval_list[i][1])
    ])

index_list = []
for i in range(0, 6):
    index_list.append(np.arange(i, i + 1.0) + 0.5)

y_dict = {
    '0': [], '1': [],
    '2': [], '3': [],
    '4': [], '5': [],
    '6': [], '7': [],
    '8': [], '9': [],
    '10': [], '11': []
}

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
def animate_two(i):
    print(i)
    plt.clf()

    if 0 <= i < 120:
        y_dict['0'], y_dict['1'] = lst_ranges[0][0][i], lst_ranges[0][1][i]
        y_dict['2'], y_dict['3'] = lst_ranges[1][0][i], lst_ranges[1][1][i]
        y_dict['4'], y_dict['5'] = lst_ranges[2][0][i], lst_ranges[2][1][i]
        y_dict['6'], y_dict['7'] = lst_ranges[3][0][i], lst_ranges[3][1][i]
        y_dict['8'], y_dict['9'] = lst_ranges[4][0][i], lst_ranges[4][1][i]
        y_dict['10'], y_dict['11'] = lst_ranges[5][0][i], lst_ranges[5][1][i]
    else:
        y_dict['0'], y_dict['1'] = lst_ranges[0][0][-1], lst_ranges[0][1][-1]
        y_dict['2'], y_dict['3'] = lst_ranges[1][0][-1], lst_ranges[1][1][-1]
        y_dict['4'], y_dict['5'] = lst_ranges[2][0][-1], lst_ranges[2][1][-1]
        y_dict['6'], y_dict['7'] = lst_ranges[3][0][-1], lst_ranges[3][1][-1]
        y_dict['8'], y_dict['9'] = lst_ranges[4][0][-1], lst_ranges[4][1][-1]
        y_dict['10'], y_dict['11'] = lst_ranges[5][0][-1], lst_ranges[5][1][-1]

    rect0 = plt.bar(index_list[0] - 0.12, y_dict['0'], bar_width, color='gray', label=rows[1])
    plt.bar_label(rect0, fmt='%1.1f', fontsize=20)
    rect1 = plt.bar(index_list[0] + 0.12, y_dict['1'], bar_width, color='green', label=rows[0])
    plt.bar_label(rect1, fmt='%1.1f', fontsize=20)
    rect2 = plt.bar(index_list[1] - 0.12, y_dict['2'], bar_width, color='gray')
    plt.bar_label(rect2, fmt='%1.1f', fontsize=20)
    rect3 = plt.bar(index_list[1] + 0.12, y_dict['3'], bar_width, color='green')
    plt.bar_label(rect3, fmt='%1.1f', fontsize=20)
    rect4 = plt.bar(index_list[2] - 0.12, y_dict['4'], bar_width, color='gray')
    plt.bar_label(rect4, fmt='%1.1f', fontsize=20)
    rect5 = plt.bar(index_list[2] + 0.12, y_dict['5'], bar_width, color='green')
    plt.bar_label(rect5, fmt='%1.1f', fontsize=20)
    rect6 = plt.bar(index_list[3] - 0.12, y_dict['6'], bar_width, color='gray')
    plt.bar_label(rect6, fmt='%1.1f', fontsize=20)
    rect7 = plt.bar(index_list[3] + 0.12, y_dict['7'], bar_width, color='green')
    plt.bar_label(rect7, fmt='%1.1f', fontsize=20)
    rect8 = plt.bar(index_list[4] - 0.12, y_dict['8'], bar_width, color='gray')
    plt.bar_label(rect8, fmt='%1.1f', fontsize=20)
    rect9 = plt.bar(index_list[4] + 0.12, y_dict['9'], bar_width, color='green')
    plt.bar_label(rect9, fmt='%1.1f', fontsize=20)
    rect10 = plt.bar(index_list[5] - 0.12, y_dict['10'], bar_width, color='gray')
    plt.bar_label(rect10, fmt='%1.1f', fontsize=20)
    rect11 = plt.bar(index_list[5] + 0.12, y_dict['11'], bar_width, color='green')
    plt.bar_label(rect11, fmt='%1.1f', fontsize=20)

    plt.legend(loc='upper center', ncols=2, fontsize=20)
    plt.ylabel(f"Время выполнения, секунды", fontdict={'fontsize': 20})
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], columns, fontsize=20)
    plt.yticks(values, fontsize=20)
    plt.title('Сравнение времени ML задач в Python и Postgres + PL/Python (меньше лучше)', fontdict={'fontsize': 20})
    plt.ylim([0, 54])
    plt.xlim([0.0, 6.0])
    plt.tight_layout()

ani = FuncAnimation(fig, animate_two, frames=240, interval=10)
ani.save('postgres-vs-python-s.mp4', writer='ffmpeg', fps=60)

