import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


def main():
    x_sunlight = np.arange(0, 11, 1)  # Nasłonecznienie [0, 10]
    x_vegetation = np.arange(0, 101, 1)  # Poziom wegetacji [0, 100]

    sunlight_low = fuzz.trimf(x_sunlight, [0, 0, 5])
    sunlight_med = fuzz.trimf(x_sunlight, [0, 5, 10])
    sunlight_high = fuzz.trimf(x_sunlight, [5, 8, 10])
    sunlight_very_high = fuzz.trimf(x_sunlight, [8, 10, 10])

    veg_low = fuzz.trimf(x_vegetation, [0, 0, 50])
    veg_med = fuzz.trimf(x_vegetation, [0, 50, 100])
    veg_high = fuzz.trimf(x_vegetation, [50, 100, 100])

    _, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))

    ax0.plot(x_sunlight, sunlight_low, 'b', linewidth=1.5, label='Niskie')
    ax0.plot(x_sunlight, sunlight_med, 'g', linewidth=1.5, label='Średnie')
    ax0.plot(x_sunlight, sunlight_high, 'r', linewidth=1.5, label='Wysokie')
    ax0.plot(x_sunlight, sunlight_very_high, 'y', linewidth=1.5, label='Bardzo wysokie')
    ax0.set_title('Nasłonecznienie')
    ax0.legend()

    ax1.plot(x_vegetation, veg_low, 'b', linewidth=1.5, label='Niski')
    ax1.plot(x_vegetation, veg_med, 'g', linewidth=1.5, label='Średni')
    ax1.plot(x_vegetation, veg_high, 'r', linewidth=1.5, label='Wysoki')
    ax1.set_title('Poziom wegetacji')
    ax1.legend()

    plt.tight_layout()
    plt.show()

    sunlight_level = 1

    sunlight_level_low = fuzz.interp_membership(x_sunlight, sunlight_low, sunlight_level)
    sunlight_level_med = fuzz.interp_membership(x_sunlight, sunlight_med, sunlight_level)
    sunlight_level_high = fuzz.interp_membership(x_sunlight, sunlight_high, sunlight_level)
    sunlight_level_very_high = fuzz.interp_membership(x_sunlight, sunlight_very_high, sunlight_level)

    # Reguła 1: Jeśli nasłonecznienie jest niskie, to poziom wegetacji jest niski
    veg_activation_low = np.fmin(sunlight_level_low, veg_low)

    # Reguła 2: Jeśli nasłonecznienie jest średnie, to poziom wegetacji jest średni
    veg_activation_med = np.fmin(sunlight_level_med, veg_med)

    # Reguła 3: Jeśli nasłonecznienie jest wysokie, to poziom wegetacji jest wysoki
    veg_activation_high = np.fmin(sunlight_level_high, veg_high)

    # Reguła 4: Jeśli nasłonecznienie jest bardzo wysokie, to poziom wegetacji jest średni
    veg_activation_very_high = np.fmin(sunlight_level_very_high, veg_med)

    veg0 = np.zeros_like(x_vegetation)

    aggregated = np.fmax(veg_activation_low,
                         np.fmax(veg_activation_med, np.fmax(veg_activation_high, veg_activation_very_high)))

    vegetation_level = fuzz.defuzz(x_vegetation, aggregated, 'centroid')
    vegetation_activation = fuzz.interp_membership(x_vegetation, aggregated, vegetation_level)

    _, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_vegetation, veg_low, 'b', linewidth=0.5, linestyle='--')
    ax0.plot(x_vegetation, veg_med, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_vegetation, veg_high, 'r', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_vegetation, veg0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([vegetation_level, vegetation_level], [0, vegetation_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Agregacja i wynik (wegetacja)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
