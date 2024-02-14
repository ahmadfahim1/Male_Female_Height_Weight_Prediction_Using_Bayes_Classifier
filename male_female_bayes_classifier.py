import numpy as np
import matplotlib.pyplot as plt
import random


male_female_X_train = np.loadtxt('male_female_X_train.txt')
male_female_X_test = np.loadtxt('male_female_X_test.txt')
male_female_y_train = np.loadtxt('male_female_y_train.txt')
male_female_y_test = np.loadtxt('male_female_y_test.txt')

height_bins = 10
weight_bins = 10
height_range = [80, 220]
weight_range = [30, 180]



male_heights = male_female_X_train[male_female_y_train == 0, 0]
female_heights = male_female_X_train[male_female_y_train == 1, 0]
male_weights = male_female_X_train[male_female_y_train == 0, 1]
female_weights = male_female_X_train[male_female_y_train == 1, 1]


male_probability = len(male_heights) / len(male_female_X_train)
female_probability = len(female_heights) / len(male_female_X_train)


male_height_hist, _ = np.histogram(male_heights, bins=height_bins, range=height_range)
female_height_hist, female_height_bins = np.histogram(female_heights, bins=height_bins, range=height_range)
male_weight_hist, _ = np.histogram(male_weights, bins=weight_bins, range=weight_range)
female_weight_hist, female_weight_bins = np.histogram(female_weights, bins=weight_bins, range=weight_range)

height_bin_centers = 0.5 * (female_height_bins[:-1] + female_height_bins[1:])
weight_bin_centers = 0.5 * (female_weight_bins[:-1] + female_weight_bins[1:])


predict_height = []
predict_weight = []
predict_height_weight = []


for data in male_female_X_test:
    height, weight = data


    height_bin = np.argmin(np.abs(height_bin_centers - height))
    probability_height = (male_height_hist[height_bin] + female_height_hist[height_bin]) / len(male_female_y_train)
    probability_height_given_male = male_height_hist[height_bin] / len(male_heights)
    probability_height_given_female = female_height_hist[height_bin] / len(female_heights)
    probability_male_given_height = (probability_height_given_male * male_probability) / probability_height
    probability_female_given_height = (probability_height_given_female * female_probability) / probability_height

    if probability_female_given_height > probability_male_given_height:
        predict_height.append(1)
    else:
        predict_height.append(0)


    weight_bin = np.argmin(np.abs(weight_bin_centers - weight))
    probability_weight = (male_weight_hist[weight_bin] + female_weight_hist[weight_bin]) / len(male_female_y_train)
    prob_weight_given_male = male_weight_hist[weight_bin] / len(male_weights)
    probability_weight_given_female = female_weight_hist[weight_bin] / len(female_weights)
    probability_male_given_weight = (prob_weight_given_male * male_probability) / probability_weight
    probability_female_given_weight = (probability_weight_given_female * female_probability) / probability_weight

    if probability_female_given_weight > probability_male_given_weight:
        predict_weight.append(1)
    else:
        predict_weight.append(0)


    probability_male = probability_male_given_height * probability_male_given_weight
    probability_female = probability_female_given_height * probability_female_given_weight

    if probability_female > probability_male:
        predict_height_weight.append(1)
    else:
        predict_height_weight.append(0)


cls_height = sum(predict_height == male_female_y_test) / len(male_female_y_test)
print(f"Accuracy based on height: {cls_height*100:.2f}%")

cls_weight = sum(predict_weight == male_female_y_test) / len(male_female_y_test)
print(f"Accuracy based on weight: {cls_weight*100:.2f}%")

cls_height_weight = sum(predict_height_weight == male_female_y_test) / len(male_female_y_test)
print(f"Accuracy based on height and weight: {cls_height_weight*100:.2f}%")
