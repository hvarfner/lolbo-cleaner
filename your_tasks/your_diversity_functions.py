''' Diversity functions useed for running ROBOT to discover
    a diverse set of optimal solutions with some minimum diversity between eachother
    Functions must take in two input space items (x1, x2)
    and return a float value indicating the diversity between them
'''
import sys 
sys.path.append("../")

def string_edit_distance(s1, s2): 
    ''' Returns Levenshtein Edit Distance btwn two strings'''
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1 
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j] 

# Diversity functions with unique string identifiers 
# identifiers can be passed in when running ROBOT to specify which diversity function to use 
DIVERSITY_FUNCTIONS_DICT = {
    'edit_dist':string_edit_distance
}
