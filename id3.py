import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
data = pd.read_csv('C:\\Users\\Kelsey\\Downloads\\CodeExerc\\codes\\fastGold.csv')
print("\n Given Data Set:\n\n", data)

#Function to calculate the entropy of probaility of observations
# -p*log2*p

def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

###################################Function to calulate the entropy of the given Data Sets/List with respect to target attributes#######################

def entropy_of_list(a_list):  
    from collections import Counter
    cnt = Counter(x for x in a_list)   # Counter calculates the propotion of class
   # print("\nClasses:",cnt)
    #print("No and Yes Classes:",a_list.name,cnt)
    num_instances = len(a_list)*1.0   # = 14
    print("\n Number of Instances of the Current Sub Class is {0}:".format(num_instances ))
    probs = [x / num_instances for x in cnt.values()]  # x means no of YES/NO
    print("\n Classes:",min(cnt),max(cnt))
    print(" \n Probabilities of Class {0} is {1}:".format(min(cnt),min(probs)))
    print(" \n Probabilities of Class {0} is {1}:".format(max(cnt),max(probs)))
    return entropy(probs) # Call Entropy :
    
# The initial entropy of the YES/NO attribute for our dataset.
print("\n  INPUT DATA SET FOR ENTROPY CALCULATION:\n", data['Fast'])

total_entropy = entropy_of_list(data['Fast'])

print("\n Total Entropy of Fast Data Set:",total_entropy)

#####################################################calculating information Gain#######################################################

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    print("Information Gain Calculation of ",split_attribute_name)
    '''
    Takes a DataFrame of attributes, and quantifies the entropy of a target
    attribute after performing a split along the values of another attribute.
    '''
    # Split Data by Possible Vals of Attribute:
    df_split = df.groupby(split_attribute_name)
    # Calculate Entropy for Target Attribute, as well as
    # Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
  
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    #if trace: # helps understand what fxn is doing:
    
    # Calculate Information Gain:
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


print('Info-gain for FuelEco is :'+str( information_gain(data, 'FuelEco', 'Fast')),"\n")
print('\n Info-gain for Engine is: ' + str( information_gain(data, 'Engine', 'Fast')),"\n")
print('\n Info-gain for Turbo is:' + str( information_gain(data, 'Turbo', 'Fast')),"\n")
print('\n Info-gain for Weight is:' + str( information_gain(data, 'Weight','Fast')),"\n")

######################################ID3 Algorithm#########################################################################################

def id3(df, target_attribute_name, attribute_names, default_class=None):
    
    ## Tally target attribute:
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])# class of YES /NO
    
    ## First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
    
    ## Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class  # Return None for Empty Data Set
    
    ## Otherwise: This dataset is ready to be devied up!
    else:
        # Get Default Value for next recursive call of this function:
        default_class = max(cnt.keys()) #No of YES and NO Class
        # Compute the Information Gain of the attributes:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] #
        index_of_max = gainz.index(max(gainz)) # Index of Best Attribute
        # Choose Best Attribute to split on:
        best_attr = attribute_names[index_of_max]
        
        # Create an empty tree, to be populated in a moment
        tree = {best_attr:{}} # Iniiate the tree with best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split dataset
 ############ ##################On each split, recursively call this algorithm.########################
                                # populate the empty tree with subtrees, which
                                     # are the result of the recursive call
                                     
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

# Get Predictor Names (all but 'class')
attribute_names = list(data.columns)
print("List of Attributes:", attribute_names) 
attribute_names.remove('Fast') #Remove the class attribute 
print("Predicting Attributes:", attribute_names)

# Run Algorithm:
from pprint import pprint
tree = id3(data,'Fast',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)
attribute = next(iter(tree))

#####################network x draw tree###############
G = nx.Graph()
nodes=["Weight","Average","Heavy","Light","Engine",
       "Turbo","Large","Small","No","Yes"," No"," Yes","No ","Yes ","yes"]

G.add_nodes_from(nodes)
G.nodes()

G.add_edge("Weight","Average")
G.add_edge("Weight","Heavy")
G.add_edge("Weight","Light")
G.add_edge("Average","Engine")
G.add_edge("Heavy","Turbo")
G.add_edge("Light","yes")
G.add_edge("Turbo"," No")
G.add_edge("Turbo"," Yes")
G.add_edge(" No","No")
G.add_edge(" Yes","Yes")
G.add_edge("Engine","Large")
G.add_edge("Engine","Small")
G.add_edge("Large","Yes ")
G.add_edge("Small","No ")

G.node["Weight"]['pos']=(0,0)
G.node["Average"]['pos']=(-3,-2)
G.node["Heavy"]['pos']=(0,-2)
G.node["Light"]['pos']=(2,-2)
G.node["Engine"]['pos']=(-3,-4)
G.node["Turbo"]['pos']=(0,-4)
G.node["yes"]['pos']=(2,-4)
G.node["Large"]['pos']=(-4,-6)
G.node["Small"]['pos']=(-2,-6)
G.node[" No"]['pos']=(-1,-6)
G.node[" Yes"]['pos']=(1,-6)
G.node["No "]['pos']=(-2,-7)
G.node["Yes "]['pos']=(-4,-7)
G.node["No"]['pos']=(-1,-7)
G.node["Yes"]['pos']=(1,-7)


node_pos = nx.get_node_attributes(G, 'pos')

nx.draw_networkx(G, node_pos, node_color='darkturquoise', node_size=450)
nx.draw_networkx_edges(G, node_pos, width=2, edge_color='peru')
plt.axis('off')
plt.show()
