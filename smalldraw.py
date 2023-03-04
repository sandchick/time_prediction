import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制完整图像
fig, ax = plt.subplots()
ax.plot(x, y)

# 创建子图，绘制局部放大的图像
ax_zoom = fig.add_axes([0.6, 0.2, 0.25, 0.25])
ax_zoom.plot(x, y)
ax_zoom.axis([2, 4, 0.5, 1])

# 显示图像
plt.show()