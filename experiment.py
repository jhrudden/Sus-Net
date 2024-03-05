from env import MazeEnv, MoveAction
import matplotlib.pyplot as plt

plt.ion()

m = MazeEnv(10, 2)


plt.show()
while True:
    m.render()

    action = input("Action: ")
    # determine action direction based on input being w, a, s, or d
    a = 0
    if action == "w":
        a = 0
    if action == "a":
        a = 2
    if action == "s":
        a = 1
    if action == "d":
        a = 3
    if action == "q":
        break
    state, reward, done, truncated, info = m.step(MoveAction(a))
    print("State: ", state)
    if done:
        break

    plt.close()
