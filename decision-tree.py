import pandas as pd 
import numpy as np
import sys
import json

def id3(train_data_m, label):
    train_data = train_data_m.copy() 
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree(tree, None, train_data, label, class_list) 
    return tree

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] == 0: 
        return

    max_info_feature = find_most_informative_feature(train_data, label, class_list) 
    tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) 
    next_root = None
        
    if prev_feature_value != None: 
        root[prev_feature_value] = dict()
        root[prev_feature_value][max_info_feature] = tree
        next_root = root[prev_feature_value][max_info_feature]
    else: 
        root[max_info_feature] = tree
        next_root = root[max_info_feature]
        
    for node, branch in list(next_root.items()): 
        if branch == "?": 
            feature_value_data = train_data[train_data[max_info_feature] == node] 
            make_tree(next_root, node, feature_value_data, label, class_list) 

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) 
    tree = {} 

    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value] 
        
        assigned_to_node = False 
        for c in class_list: 
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] 

            if class_count == count: 
                tree[feature_value] = c 
                train_data = train_data[train_data[feature_name] != feature_value] 
                assigned_to_node = True

        if not assigned_to_node: 
            tree[feature_value] = "?" 
            
    return tree, train_data

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)

    max_info_gain = -1
    max_info_feature_name = None

    for feature in feature_list: 
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature_name = feature

    return max_info_feature_name


def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row  = train_data.shape[0]

    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data,label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy

    return calc_total_entropy(train_data, label,class_list) - feature_info

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0

    for c in class_list: 
        total_class_count = train_data[train_data[label] == c].shape[0] 
        total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count/ total_row)

        total_entr += total_class_entr

    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]

    entropy = 0

    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count == 0:
            continue

        probability_class = label_class_count / class_count
        entropy_class = - probability_class * np.log2(probability_class) 

        entropy += entropy_class

    return entropy 

def walk_decision_tree(tree): 
    if tree is None:
        return

    if isinstance(tree, str):
       print(f"Sizin nəticəniz: {resultLangMappings[tree]}")
       return

    for en, _ in tree.items(): 
      question = questions[en]
      print(question)

      ans = input("Sizin cavabınız: ")

      if ans in reversed_mappings:
        translation = reversed_mappings[ans]

        walk_decision_tree(tree[en][translation])
      else:
        walk_decision_tree(tree[en][ans])

train_data_m = pd.read_csv("StudentGrades.csv")
# calc_total_entropy(train_data_m, "Next Course Eligibility", ["Yes", "No"])
# print(calc_info_gain("Learning Style", train_data_m, "Next Course Eligibility", ["Yes", "No"]))
# print(calc_entropy(train_data_m[train_data_m["Learning Style"] == "Visual"], "Next Course Eligibility", ["Yes", "No"]))



tree = id3(train_data_m, "Next Course Eligibility")
print(json.dumps(tree))

# print("Süni intellekt ilkin məlumat mənbəsini proses etdi! Suallara cavab verərək Adaptiv təhsilin sizin üçün seçkdiyi nəticələri görə bilərsiz")
# Yuxarıdakı model ümumi Decision Tree algoritmidir və mümkün bütün məlumat çoxluqları keçərlidir

questions = {
    "Previous Course Difficulty": "Əvvəlki kurs sizə nə qədər çətin gəlmişdir? Aşağıdakılardan birini seçin\nÇox Asan\nAsan\nOrta\nÇətin\nÇox Çətin",
    "Learning Style": "Hansı tədris metodu sizə daha uyğundur? Aşağıdakılardan birini seçin\nVizual\nİnteraktiv\nOxuma/Yazma\nDinləmə",
    "Previous Course Grade": "Bundan əvvəlki kursun nəticəsi verilənlərdən hansıdır? A,B,C,D,E variantlarından birini yazın",
    "Group Project Involvement": "Əvvəlki kurs ərzində keçirilən proyektdə işitirakını aşağıdakılardan biri ilə qiymətləndirin? \nAşağı, Yuxarı",
    "Motivation Level": "Motivasiya səviyyənizi aşağıdakılardan biri ilə qiymətləndirin? \nAşağı, Orta, Yuxarı, Çox Yuxarı",
}
resultLangMappings = {
    "Medium" : "Orta",
    "Easy" : "Asan",
    "Very Easy": "Çox Asan",
    "Moderate" : "Orta",
    "Difficult" : "Çətin",
    "Low" : "Aşağı",
    "High" : "Yuxarı",
    "Very High": "Çox Yuxarı",
    "Visual" : "Vizual",
    "Kinesthetic": "Interaktiv",
    "Read/Write" : "Oxuma/Yazma",
    "Auditory" : "Dinləmə",
    "Yes": "Növbəti kursa uyğun deyil",
    "No": "Növbəti kursa uyğundur"
}

reversed_mappings = {v: k for k, v in resultLangMappings.items()}

# questions = {
#     "Əvvəlki kursun çətinlik səviyyəsi": "Əvvəlki kurs sizə nə qədər çətin gəlmişdir? Aşağıdakılardan birini seçin\nÇox Asan\nAsan\nOrta\nÇətin\nÇox Çətin",
#     "Tədris metodu,": "Hansı tədris metodu sizə daha uyğundur? Aşağıdakılardan birini seçin\nVizual\nİnteraktiv\nOxuma/Yazma\nDinləmə",
#     "Əvvəlki kursun nəticəsi": "Bundan əvvəlki kursun nəticəsi verilənlərdən hansıdır? A,B,C,D,E variantlarından birini yazın",
#     "Əvvəlki proyektdə iştirak": "Əvvəlki kurs ərzində keçirilən proyektdə işitirakını aşağıdakılardan biri ilə qiymətləndirin? \nAşağı, Yuxarı",
#     "Motivasiya səviyyəsi,": "Motivasiya səviyyənizi aşağıdakılardan biri ilə qiymətləndirin? \nAşağı, Orta, Yuxarı, Çox Yuxarı",
# }


# walk_decision_tree(tree)