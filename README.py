# recom-system
to recomend product

# In[1]:

# define the number of products in our 'database'

num_products = 10


# In[2]:

# define the number of users in our 'database'

num_users = 5


# In[3]:

# randomly initialize some product ratings 
# a 10 X 5 matrix

ratings = random.randint(11, size = (num_products, num_users))


# In[4]:

print ratings


# In[5]:

# create a logical matrix (matrix that represents whether a rating was made, or not)
# != is the logical not operator

did_rate = (ratings != 0) * 1


# In[6]:

print did_rate


# In[7]:

# Here's what happens if we don't multiply by 1

print (ratings != 0)


# In[8]:

print (ratings != 0) * 1


# In[9]:

# Get the dimensions of a matrix using the shape property

ratings.shape


# In[10]:

did_rate.shape


# In[11]:

# Let's make some ratings. A 10 X 1 column vector to store all the ratings I make

aadei_ratings = zeros((num_products, 1))
print aadei_ratings


# In[12]:

# Python data structures are 0 based

print aadei_ratings[10] 


# In[13]:

# I rate 3 movies

aadei_ratings[0] = 8
aadei_ratings[4] = 7
aadei_ratings[7] = 3

print aadei_ratings


# In[14]:

# Update ratings and did_rate

ratings = append(aadei_ratings, ratings, axis = 1)
did_rate = append(((aadei_ratings != 0) * 1), did_rate, axis = 1)


# In[15]:

print ratings


# In[16]:



ratings.shape


# In[17]:

did_rate


# In[18]:

print did_rate


# In[19]:

did_rate.shape


# In[20]:

# Simple explanation of what it means to normalize a dataset

a = [10, 20, 30]
aSum = sum(a)


# In[21]:

print aSum


# In[22]:

aMean = aSum / 3


# In[23]:

print aMean


# In[24]:

aMean = mean(a)
print aMean


# In[25]:

a = [10 - aMean, 20 - aMean, 30 - aMean]
print a


# In[26]:

print ratings


# In[27]:

# a function that normalizes a dataset

def normalize_ratings(ratings, did_rate):
    num_products = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_products, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_product): 
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith product only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

        
    


# In[28]:

# Normalize ratings

ratings, ratings_mean = normalize_ratings(ratings, did_rate)


# In[29]:

# Update some key variables now

num_users = ratings.shape[1]
num_features = 3


# In[30]:

# Simple explanation of what it means to 'vectorize' a linear regression

X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])


# In[31]:

print X


# In[32]:

print Theta


# In[33]:

Y = X.dot(Theta)
print Y


# In[34]:

# Initialize Parameters theta (user_prefs), X (movie_features)

product_features = random.randn( num_products, num_features )
user_prefs = random.randn( num_users, num_features )
initial_X_and_theta = r_[product_features.T.flatten(), user_prefs.T.flatten()]


# In[35]:

print product_features


# In[36]:

print user_prefs


# In[37]:

print initial_X_and_theta


# In[38]:

initial_X_and_theta.shape


# In[39]:

product_features.T.flatten().shape


# In[40]:

user_prefs.T.flatten().shape


# In[41]:

initial_X_and_theta


# In[42]:

def unroll_params(X_and_theta, num_users, num_products, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_products * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_products)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_products * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta


# In[43]:

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_products, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_products, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# In[44]:

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_products, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_products, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization


# In[45]:

# import these for advanced optimizations (like gradient descent)

from scipy import optimize


# In[46]:

# regularization paramater

reg_param = 30


# In[47]:

# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, 								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True ) 


# In[ 48]:

cost, optimal_product_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


# In[ 49]:

# unroll once again

product_features, user_prefs = unroll_params(optimal_product_features_and_user_prefs, num_users, num_products, num_features)


# In[ 50]:

print product_features


# In[51 ]:

print user_prefs


# In[ 52]:

# Make some predictions (product recommendations). Dot product

all_predictions = product_features.dot( user_prefs.T )


# In[53 ]:

print all_predictions


# In[ 54]:

# add back the ratings_mean column vector to my (our) predictions

predictions_for_aadei = all_predictions[:, 0:1] + ratings_mean


# In[55 ]:

print predictions_for_aadei


# In[ 56]:
