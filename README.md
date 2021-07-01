
![Header Image]()

# Simple Regression (Tensorflow)
## Description
This is a simple program developed using the **Tensorflow** framework. This is program has the intention to cover the basic functionalities of how **Neural Networks** behaviors when analyzing data from different arrays. In addition, it is noticeable the usage of libraries such as Pandas, NumPy, and MatPlotLib to create array examples and visual representations of the calculation and analysis done by the AI. In this program, 2 matrices are created using *random library* to represent real datasets. Those matrices are then calculated and the program can create a linear relationship between the results and represent them by using a scatterplot graph. Furthermore, a correlation line is implemented by the usage of the equation *Y = MX + B* where *X* is a placeholder input hence defining the value of *Y*.

## Screenshot
![Game Running]()

## What have I learned
Creating this game enabled me to practice a few aspects:
* Practice Python coding skills and good practices.
* Get familiar with Pandas, Numpy and MatPlotLib.
* Develop Deep Learning skills.
* Develop Tensorflow affinity.
* Get familiar with conditions, loops, matrix, arrays, and graph representation.

## Code highlight
This block of code was able to enhance the result given by the Neural Network. When changing the learning interval, the AI will be more effective, as it will train more to improve the quality of the calculation.

```
with tf.Session() as sess:
    
    sess.run(init)
    
    training_steps = 100 #learning interval
    
    for i in range(training_steps):
        
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m, b])
```


> This is a companion project to Python 3.8 Full Stack Masterclass, check out the full course at www.udemy.com


![Footer Image]()
