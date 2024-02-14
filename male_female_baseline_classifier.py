import matplotlib.pyplot as plt
import numpy as np
import random

male_female_X_train = 'male_female_X_train.txt'
male_female_y_train = 'male_female_y_train.txt'
male_female_X_test = 'male_female_X_test.txt'
male_female_y_test = 'male_female_y_test.txt'

with open(male_female_X_train, 'r',encoding ="utf8") as f:
     male_female_height_weight = [x.rstrip().split(' ') for x in f.readlines()]

male_female_height_weight = np.array(male_female_height_weight)

with open(male_female_y_train, 'r',encoding ="utf8") as f:
    male_female_class = [x for x in f.readlines()]

male_female_class = np.array(male_female_class)
male_female_class = male_female_class.astype(float)

#length = 2389
random_male_female_list = [random.randint(0, 1) for _ in range(len(male_female_height_weight))]

male_female_ran = np.array(random_male_female_list)
# Print the list
#print(male_female_ran)

with open(male_female_y_test, 'r',encoding ="utf8") as f:
    male_female_test_class = [x for x in f.readlines()]

male_female_test_class = male_female_class.astype(int)
male_female_test_class_int = np.asarray(male_female_class, dtype=int)
#male_female_test_class_int

correct_prediction = 0

for i in range (0, len(male_female_test_class_int)):
    if male_female_ran[i] == male_female_test_class_int[i]:
        correct_prediction += 1

accuracy = correct_prediction/len(male_female_test_class_int)
print(f"Accuracy_Random = {accuracy*100:.2f}%")


print("Output for baseline classifier 2")

counter_male = 0
counter_female = 0

for i in range(0,len(male_female_test_class_int)):
    if male_female_test_class_int[i] == 0:
        counter_male += 1
    else:
        counter_female += 1

    
#print(f"Male = {counter_male}, Female = {counter_female}")
#print(f"Total Test Data: {len(male_female_test_class_int)}")

new_array = male_female_test_class_int

new_class = np.where(new_array == 0, 1, new_array)

correct_n_class_prediction = 0

for i in range (0, len(male_female_test_class_int)):
    if new_class[i] == male_female_test_class_int[i]:
        correct_n_class_prediction += 1

accuracy = correct_n_class_prediction/len(male_female_test_class_int)
print(f"Accuracy_highest_priory = {accuracy*100:.2f}%")


