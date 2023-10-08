#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Function to generate and display Fibonacci series up to 10 terms
def fibonacci(n):
    fib_series = []
    a, b = 0, 1

    for _ in range(n):
        fib_series.append(a)
        a, b = b, a + b

    return fib_series

# Display the Fibonacci series up to 10 terms
fib_terms = fibonacci(10)
print("Fibonacci Series up to 10 terms:")
for term in fib_terms:
    print(term)

    


# In[4]:


# Sample list
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Display numbers at odd indices
print("Numbers at odd indices:")
for i in range(1, len(my_list), 2):
    print(my_list[i])

    


# In[5]:


# Sample list
my_list = [1, 2, 3, 4, 5]

# Print the list in reverse order
print("List in reverse order:", my_list[::-1])



# In[3]:


string = """
	ChatGPT has created this text to provide tips on creating interesting paragraphs. 
	First, start with a clear topic sentence that introduces the main idea. 
	Then, support the topic sentence with specific details, examples, and evidence.
	Vary the sentence length and structure to keep the reader engaged.
	Finally, end with a strong concluding sentence that summarizes the main points.
	Remember, practice makes perfect!
	"""

st = string.replace(',', '').replace('.', '').replace('!', '').replace('\n', '').strip().split()

print(len(set(st)))







# In[4]:


def count_vowels(word):
    # Define a set of vowels
    vowels = set("AEIOUaeiou")
    
    # Initialize a variable to store the count of vowels
    vowel_count = 0
    
    # Iterate through each character in the word
    for char in word:
        # Check if the character is a vowel
        if char in vowels:
            vowel_count += 1
    
    return vowel_count

# Example usage:
word = "Hello"
result = count_vowels(word)
print(f"The word '{word}' contains {result} vowels.")


# In[5]:


animals = ['tiger', 'elephant', 'monkey', 'zebra', 'panther']

for animal in animals:
    print(animal.upper())

    


# In[6]:


for num in range(1, 16):
    if num % 2 == 0:
        print(f"{num} is even")
    else:
        print(f"{num} is odd")


# In[7]:


# Input: Take two integers as input from the user
num1 = int(input("Enter the first integer: "))
num2 = int(input("Enter the second integer: "))

# Calculate the sum of the two integers
sum_of_integers = num1 + num2

# Output: Return the sum
print(f"The sum of {num1} and {num2} is: {sum_of_integers}")


# In[ ]:




