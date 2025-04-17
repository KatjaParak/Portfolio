import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_file():
    cardio = pd.read_csv("cardio_train.csv", sep=';')
    return cardio


def calculate_bmi(cardio):
    bmi = np.round(np.divide(cardio['weight'],
                   np.power(cardio['height']*0.01, 2)))
    return bmi


def remove_outliers(cardio, feature):
    q3 = np.quantile(cardio[feature], 0.75)
    q1 = np.quantile(cardio[feature], 0.25)
    IQR = q3 - q1
    return cardio[(cardio[feature] <= (q3 + 3 * IQR)) & (cardio[feature] >= (q1 - 3 * IQR))]


def create_bmi_class(cardio):
    return pd.cut(x=cardio['bmi'], bins=[18.5, 25, 30, 35, 40, 50],
                  labels=['normal range', 'overweight', 'obese(class I)', 'obese (class II)', 'obese (class III)'])


def create_dataset(filtered_cardio):
    new_cardio = filtered_cardio.copy()
    cardio_1 = new_cardio.drop(
        ['ap_hi', 'ap_lo', 'height', 'weight', 'bmi'], axis=1)
    cardio_2 = new_cardio.drop(
        ['bmi_class', 'pressure_category', 'height', 'weight'], axis=1)

    return pd.get_dummies(cardio_1, columns=['bmi_class', 'pressure_category', 'gender'], drop_first=True), pd.get_dummies(cardio_2, columns=['gender'], drop_first=True)

# the code was sourced from statology.org


def set_pressure_category(cardio):
    conditions = [
        (cardio['ap_hi'] >= 90) & (cardio['ap_hi'] <= 120) & (
            cardio['ap_lo'] <= 80) & (cardio['ap_lo'] >= 60),
        ((cardio['ap_hi'] >= 120) & (cardio['ap_hi'] <= 129)) & (
            cardio['ap_lo'] <= 80) & (cardio['ap_lo'] >= 60),
        ((cardio['ap_hi'] >= 130) & (cardio['ap_hi'] <= 139)) & (
            (cardio['ap_lo'] <= 90) & (cardio['ap_lo'] >= 80)),
        ((cardio['ap_hi'] >= 140) & (cardio['ap_hi'] <= 180)) & (
            cardio['ap_lo'] >= 90),
        ((cardio['ap_hi'] >= 180) & (cardio['ap_hi'] <= 200)) & (
            cardio['ap_lo'] >= 120)
    ]

    category = ['healthy', 'elevated', 'stage_1_hyper',
                'stage_2_hyper', 'hyper_crisis']
    cardio['pressure_category'] = np.select(conditions, category)

    filtered_cardio = cardio[cardio['pressure_category'] != '0']
    return filtered_cardio


def plot_eda_pie(cardio):
    fig, axes = plt.subplots(2, 2, dpi=200, figsize=(18, 9))
    axes = axes.flatten()

    dataframes = [cardio['cardio'].value_counts(), cardio['cholesterol'].value_counts(),
                  cardio['smoke'].value_counts(), cardio[cardio['cardio'] == 1]['gender'].value_counts()]
    labels = [['presence', 'absence'], ['normal', 'above normal',
                                        'well above normal'], ['Non-smokers', 'smokers'], ['women', 'men']]
    colors = [['darkslategray', 'powderblue'], ['darkslategrey', 'cadetblue', 'powderblue'],
              ['darkslategray', 'powderblue'], ['darkslategray', 'powderblue']]
    explodes = [[0.02, 0.02], [0.02, 0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
    titles = ["Presence or absence of cardiovascular disease in patients", "Cholesterol levels in patients",
              "Proportion of smokers in patients", "Gender distribution of positively diagnosed patients"]

    for i, (data, label, color, explode, title) in enumerate(zip(dataframes, labels, colors, explodes, titles)):
        axes[i].pie(data, autopct='%1.2f%%', labels=label,
                    colors=color, explode=explode)
        axes[i].set(title=title)

    plt.tight_layout()
    plt.show()
    return fig


def plot_eda_hist(cardio):
    fig, axes = plt.subplots(1, 3, dpi=200, figsize=(18, 6))
    axes = axes.flatten()

    dataframes = [[i/365 for i in cardio['age']],
                  cardio['weight'], cardio['height']]
    colors = ['powderblue', 'darkslategrey', 'cadetblue']

    titles = ["Age distribution of patients",
              "Weight distribution of patients", "Height distribution of patients"]

    for i, (data, color, title) in enumerate(zip(dataframes, colors, titles)):
        sns.histplot(data, bins=100, color=color, ax=axes[i])
        axes[i].set(title=title)

    plt.tight_layout()
    plt.show()
    return fig


def plot_cardio_subplots(filtered_cardio):
    fig, axes = plt.subplots(3, 2, dpi=200, figsize=(18, 12))
    axes = axes.flatten()
    hues = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi_class']
    palettes = [['darkslategrey', 'lightgreen', 'cadetblue'], ['darkslategrey', 'powderblue', 'cadetblue'], ['darkslategrey', 'powderblue'],
                ['darkslategrey', 'cadetblue'], ['powderblue', 'darkslategrey'], ['teal', 'darkslategrey', 'cadetblue', 'powderblue', 'lightgreen']]
    titles = ["Cholesterol levels in patients with diagnosed cardiovascular diseases (CVDs)", "Glucose levels in patients with diagnosed cardiovascular diseases (CVDs)",
              "Proportion of smokers and non-smokers in CVDs-diagnosed patients", "Alcohol consumption among patients diagnosed with CVDs",
              "Physical activity levels in patients diagnosed with CVDs", "BMI classes among CVDs-diagnosed patients"]
    legends = [['normal', 'above normal', 'well above normal'], ['normal', 'above normal', 'well above normal'], ['Non-smoker', 'Smoker'], ['Non-consumer', 'Consumer'],
               ['Non-active', 'Active'], ['Normal', 'Overweight', 'Obesity(class I)', 'Obesity(class II)', 'Obesity(class III)']]

    for i, (hue, palette, title, legend) in enumerate(zip(hues, palettes, titles, legends)):
        sns.countplot(filtered_cardio[filtered_cardio['cardio'] == 1],
                      x='cardio', hue=hue, palette=palette, ax=axes[i])
        axes[i].set(title=title)
        axes[i].legend(labels=legend)

        for j in axes[i].containers:
            axes[i].bar_label(j,)

    plt.tight_layout()
    plt.show()
    return fig


def plot_corr_matrix(filtered_cardio):
    filtered_cardio = pd.get_dummies(filtered_cardio).corr()
    plt.figure(figsize=(10, 8))
    fig = sns.heatmap(filtered_cardio, linewidths=.2, annot=True, annot_kws={
                      'size': 6}, fmt='.1f', cmap='crest')
    plt.title("Correlation coefficients")
    return fig
