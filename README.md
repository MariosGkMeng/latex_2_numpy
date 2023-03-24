# ❓ What?
A Python module that converts Latex equations to numpy.

# 🤔 Why?

1. It is easier to write and **review** a scientific-computing algorithm using the original mathematical syntax, rather than the computer programming expression
2. It can be linked to scientific documentation and even publications, without the need to program something from the beginning
3. It can eliminate the need to learn programming-language specific rules (why should you know which module is needed to use linear algebra operations?)
    - With the advent of multidisciplinary collaboration, requiring time to learn programming-language-specific knowledge creates delays and frustration for people who wish to concentrate on the beauty of their science and collaborate with others
5. Since it is easier to read (equations do not appear in the computer-programming format), it can make scientific computing more fun and less tiring

# 🎯 Goal
The ultimate goal for this tool is to be an autonomous programming language, using Python as a sort of interpreter. The scientific code will be written in a markdown-latex format (in [Obsidian](https://obsidian.md/)) and at the same time it will be compiled in Python, which will then run the computations in the background and output them to the Obsidian interface.

Sidenode: The language performing the computations in the background does not need to be Python; it can be anything else that is capable of performing the computations.


# Example
A small computational example from the field of Reinforcement Learning can be found at the image below
See the python code (left), compared to the math format on the right.

<img width="727" alt="image" src="https://user-images.githubusercontent.com/61937432/227475003-9d2ccb3d-7687-477a-8f4f-39bd6dba26d3.png">

# 👩‍💻👨‍💻📅 General Roadmap

1. Obsidian to python-numpy converter (ongoing)
2. python-outputs communicator to Obsidian
3. error-handling module in Obsidian
    - Even better: intelligent error-correction for Obsidian-coding
