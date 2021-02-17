# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:21:37 2021

@author: jjsch
"""
# Decision Making With Matrices
import numpy as np
from scipy.stats import rankdata
# This is a pretty simple assignment.  You will do something you do everyday, but 
#today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch. 
#You have to pick a restaurant that’s best for everyone.  Then you should decided if you should split into two groups so everyone is happier.  

# Despite the simplicity of the process you will need to make decisions
#regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.  

# You asked your 10 work friends to answer a survey. They gave you back the following 
#dictionary object.  
people = {'Jane': {'willingness to travel': 5,
                  'desire for new experience':3,
                  'cost':2,
                  'indian food':1,
                  'hipster points':1,
                  'vegetarian': 1,
                  },
          'Bob': {'willingness to travel': 3,
                  'desire for new experience':5,
                  'cost':5,
                  'indian food':1,
                  'hipster points':4,
                  'vegetarian': 1,
                  },
          'Mandy': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,
                  'indian food':4,
                  'hipster points':4,
                  'vegetarian': 2,
                  },
          'Lauren': {'willingness to travel': 5,
                  'desire for new experience':1,
                  'cost':5,
                  'indian food':2,
                  'hipster points':2,
                  'vegetarian': 1,
                  },
          'Emma': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,
                  'indian food':4,
                  'hipster points':4,
                  'vegetarian': 2,
                  },
          'Byron': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,
                  'indian food':1,
                  'hipster points':5,
                  'vegetarian': 4,
                  },
          'Kate': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':3,
                  'indian food':1,
                  'hipster points': 5,
                  'vegetarian': 4,
                  },
          'April': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':3,
                  'indian food':1,
                  'hipster points':5,
                  'vegetarian': 2,
                  },
          'Aaron': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,
                  'indian food':2,
                  'hipster points':5,
                  'vegetarian': 5,
                  },
          'Joe': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':3,
                  'indian food':4,
                  'hipster points':5,
                  'vegetarian': 3,
                  }
}

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   

orderedNames = ['Jane', 'Bob', 'Mandy', 'Lauren', 'Emma', 'Byron', 'Kate', 'April', 'Aaron','Joe']
dictlist =[]
for key in people:
    print(key)
    print(people[key].values())
    temp = list(people[key].values())
    dictlist.append(temp)

peopleMatrix = np.array(dictlist)    
print(peopleMatrix)
# Next you collected data from an internet website. You got the following information.
#You'll then create another dictionary of reviews!
restaurants  = {'Gringos':{'distance' : 2,
                          'novelty' : 2,
                          'cost': 3,
                          'average rating': 4,
                          'cuisine': 5,
                          'vegetarian': 2
                          },
                'Tapped':{'distance' : 1,
                        'novelty' : 1,
                        'cost': 5,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 2
                          },
                'Rudys':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 3,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 2
                          },
                'Corkscrew':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 3,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 1
                          },
                'StarPizza':{'distance' : 1,
                        'novelty' : 5,
                        'cost': 4,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 3
                          },
                'Redfish':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 5,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 4
                          },
                'Walkons':{'distance' : 1,
                        'novelty' : 4,
                        'cost': 5,
                        'average rating': 4,
                        'cuisine': 5,
                        'vegetarian': 4
                          },
                'Shogun':{'distance' : 5,
                        'novelty' : 3,
                        'cost': 5,
                        'average rating': 3,
                        'cuisine': 5,
                        'vegetarian': 4
                          },
                'Shilecanis':{'distance' : 4,
                        'novelty' : 4,
                        'cost': 5,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 4
                          },
                'Butlerhouse':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 5,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 2
                          },
                          
}

orderedRests = ['Gringos', 'Tapped', 'Rudys', 'Corkscrew', 'StarPizza', 'Redfish', 'Walkons', 'Shogun', 'Shilecanis', 'Butlerhouse']

def get_real_rank(data):
    return rankdata(len(data-rankdata(data)))
    
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
dictlist =[]
for key in restaurants:
    print(key)
    print(restaurants[key].values())
    temp = list(restaurants[key].values())
    dictlist.append(temp)

restMatrix = np.array(dictlist)    
print(restMatrix)

# The most important idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.




# Choose a person and compute(using a linear combination) the top restaurant for them.  
#What does each entry in the resulting vector represent? 

laurencomb = np.dot(peopleMatrix[3], restMatrix.T)
firstwinner = np.argmax(laurencomb)
winner = np.argwhere(laurencomb == np.amax(laurencomb))
print(winner)
print(orderedRests[firstwinner])
#laurentest = get_real_rank(np.dot(peopleMatrix[3], restMatrix.T))



# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  
#What does the a_ij matrix represent? 
linecomb = np.dot(peopleMatrix, restMatrix.T)

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  
#What do the entries represent?

favforall =sum(np.dot(peopleMatrix.T, restMatrix))
allwinner = firstwinner = np.argmax(favforall)
print(orderedRests[allwinner])
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  

# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?

# How should you preprocess your data to remove this problem. 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?

# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 




  
