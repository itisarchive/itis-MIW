import matplotlib.pyplot as plt
import numpy as np


def main():
    states = ['K', 'P', 'N']
    opponent_prob = [0.3, 0.4, 0.3]
    results = [[0, 1, -1],
               [-1, 0, 1],
               [1, -1, 0]]
    transition_matrix = np.array([[8, 4, 2],
                                  [2, 2, 3],
                                  [3, 2, 2]])

    state = np.random.choice(states)
    outcome = 0
    outcomes = []
    n = 20

    plt.ion()
    plt.figure()
    plt.show()

    for i in range(1, n + 1):
        opponent_choice = np.random.choice(states, p=opponent_prob)
        prediction_probs = transition_matrix[states.index(state)] / np.sum(transition_matrix[states.index(state)])
        prediction = np.random.choice(states, p=prediction_probs)
        response = states[results[states.index(prediction)].index(1)]
        result = results[states.index(opponent_choice)][states.index(response)]
        outcome += result
        outcomes.append(outcome)
        transition_matrix[states.index(state)][states.index(opponent_choice)] += 1
        state = opponent_choice

        print(f"Gra {i}\n"
              f"Przeciwnik zagrał {opponent_choice}\n"
              f"Zgadywałem, że zagra {prediction}\n"
              f"Zagrałem {response}\n"
              f"Wygrałem do kasy {result}\n"
              f"Macierz przejść\n"
              f"W kasie {outcome}\n"
              f"{transition_matrix}\n\n")

        plt.clf()
        plt.step(range(1, i + 1), outcomes, where='post')
        plt.xlabel("Iteracje")
        plt.ylabel("Wypłata")
        plt.xticks(range(0, i + 1, 2))

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
