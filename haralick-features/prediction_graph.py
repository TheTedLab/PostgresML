import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

data = [ 0.00000, 0.04146, 0.00000, 98.81749, 0.00000, 0.00062, 0.00030, 1.14012 ]

values = np.arange(0.0, 101.0, 5.0)
columns = [1, 2, 3, 4, 5, 6, 7, 8]
index = [0.166, 1.166, 2.166, 3.166, 4.166, 5.166, 6.166, 7.166]
width = 0.5
y_offset = np.zeros(len(columns))
colors = ['gray' for _ in range(7)]
max_index = data.index(max(data))
colors.insert(max_index, 'green')
rows = ['Предсказанный класс', 'Остальные классы']

plt.figure(figsize=(10, 10), dpi=100)
rects_0 = plt.bar(index, data, width, align='center', color=colors)
plt.bar_label(rects_0, labels=[f'{x:,.2f}%' for x in rects_0.datavalues], fontsize=20)
legend_colors = {'Предсказанный класс': 'green', 'Остальные классы': 'gray'}
labels = list(legend_colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=legend_colors[label]) for label in labels]
plt.legend(handles, labels, loc='upper right', ncols=1, fontsize=20)

plt.ylabel(f"Вероятность выбора класса, процент", fontdict={'fontsize': 20})
plt.xlabel(f"Метки классов", fontdict={'fontsize': 20})
plt.yticks(values, fontsize=20)
plt.xticks([0.166, 1.166, 2.166, 3.166, 4.166, 5.166, 6.166, 7.166], columns, fontsize=20)
plt.title('Результат распознавания', fontdict={'fontsize': 20})
plt.tight_layout()
plt.savefig('graphs/class_predict_2.png', bbox_inches='tight')
plt.show()

plt.clf()
