#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))


# your code here
fig, ax = plt.subplots()
bottom = np.zeros(3)

names = ["Farrah", "Fred", "Felicia"]
fruits = ['apples', 'bananas', 'oranges', 'peaches']

width=0.5

colors = {
    0: 'red',
    1: 'yellow',
    2: '#ff8000',
    3: '#ffe5b4'
}

for i in range(len(fruit)):
  label = fruits[i]
  val = fruit[i]
  ax.bar(names, val, bottom=bottom, label=label, color=colors[i], width=width)
  bottom += val

ax.set_yticks(range(0, 81, 10))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend(loc="upper right")
plt.show()
