# module "impro.py"
# Generic functions to analyse data in terms of interruptive generation (IGs) 
# or more generaly in terms of timestamps, as obtained in Collective Free Improvisation experiments
#
# created by Nicolas B. Garnier on 2022/03/05.
# Copyright 2022 ENS-Lyon - CNRS. All rights reserved.
# 
# 2022-03-24 : initial version; some functions may soon be rewritten!
# 2022-03-30 : now probing causal relationships (+ bug fixes)
#
import numpy


def find_IG(x, threshold=5):
    ''' function to get timestamps at which IGs occur in a signal x
     x         : full signal (from the pedal or slider) (assumed sampling frequency is 1Hz)
     threshold : minimal variation in the pedal signal that ensures there is an IG
    
     returned  : the timestamps of the IGs
    '''
    y=numpy.diff(x)
    if (threshold>0):
        ind = (numpy.where(y>=threshold))
    else:
        ind = (numpy.where(y<=threshold))
# https://stackoverflow.com/questions/33747908/output-of-numpy-wherecondition-is-not-an-array-but-a-tuple-of-arrays-why
    return (ind[0]+1)



# function to get the time intervals between 2 consecutive IGs
#
# x         : signal (from the pedal)
# threshold : minimal variation in the pedal signal that ensures there is an IG
# first_IG_as_interval : if ==1, then the first IG defines the end of a first segment
# 
# returned  : the time intervals between consecutive IGs
def find_IG_interval(x, threshold=5, first_IG_as_interval=1):
    IG=find_IG(x, threshold=threshold)
    intervals=numpy.diff(IG)
    if (first_IG_as_interval==1):
        intervals=numpy.insert(intervals, 0, IG[0])
    return intervals

    

# function to "clean" a set of timestamps by removing redundant values
def clean_IG(ind_x, tau=0):
    ''' function to clean a set of timestamps by removing timestamps which may be undistinguishable
     ind_x     : set of timestamps (e.g., IGs)
     tau       : minimal time between two timestamp for them to be considered distinct
                 (if tau==0, then only perfectly identical timestamps are removed)
     returned  : the cleaned set of timestamps
    '''
    if (ind_x.shape[0]>1):
        y = numpy.diff(ind_x)
        if (numpy.min(y)>tau):
            return(ind_x)
        else:
            ind = numpy.where(y<=tau)
#        print("found redundant indice", ind[0]+1)
            new = numpy.delete(ind_x, ind[0]+1)
            return(clean_IG(new, tau=tau))
    else:
        return(ind_x)



# function to count IGs from 2 musicians x,y that occur less than tau seconds appart one from the other:
#
# ind_x       : timestamps of IGs of first musician ("x")
# ind_y       : timestamps of IGs of first musician ("y")
# tau         : timescale over which we compute the matches
# do_clean    : if ==1: timestamps distant by less than \tau are removed
#               if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
# causal      : if ==0: only the time-distance matters
#               if ==1: only search for x->y "directional" matches
# returned    : the timestamps (from the ind_y timestamps series) where a match has been found
def find_IG_duo(ind_x, ind_y, tau=2, do_clean=1, causal=0):
    ''' function to count IGs from 2 musicians x,y that occur less than tau seconds appart one from the other:
     ind_x       : timestamps of IGs of first musician ("x")
     ind_y       : timestamps of IGs of first musician ("y")
     tau         : timescale over which we compute the matches
     do_clean    : if ==1: timestamps distant by less than \tau are removed
                   if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
     causal      : if ==0: only the time-distance matters
                   if ==1: only search for x->y "directional" matches
     returned    : the timestamps (from the ind_y timestamps series)

    '''
    found = numpy.array([], dtype=int)
    
    for i in ind_x:
        if (causal==0):         # first version
            ind = numpy.where(numpy.abs(ind_y-i)<=tau)
        else:                   # added 2022-03-30
            ind = numpy.where( (i<=ind_y) & ((ind_y-i)<=tau) )
#        print(i, ind_y[ind])
        found = numpy.append(found, ind_y[ind])    

    found = numpy.sort(found)   # added 2022-03-30, maybe useless
    if (do_clean>0): return clean_IG(numpy.sort(found), tau=tau)
    else:            return clean_IG(numpy.sort(found), tau=0)




# function to count the (total) number of duets as "simultaneous" timestamps (eg, IGs) in a trio
#
# ts1, ts2, ts3 : all trio timestamps (eg, IGs, or anything else)
# tau           : timescale over which we compute the matches
# bidirectional : if set to 1, then look at x->y and y->x       # deprecated 2022-03-30
# causal        : if ==0: then look at causal (or directional) interactions
# do_clean      : if ==1: timestamps distant by less than \tau are removed
#                 if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
# do_fraction   : if ==0: returns the numbers
#                 if ==1: returns the fractions
#
# returned : the total number of duos found, so
# - divided by 2(=2!) if bi-directional==1 (default)
# - divided by 6(=3!) if output is such that do_fraction==1
def count_duo_matches_in_trio(ts1, ts2, ts3, tau=2, causal=0, do_clean=1, do_fraction=0):
    ''' function to count the (total) number of "simultaneous" timestamps (eg, IGs) in a trio
     -> this function os for "duets"
     
     ts1, ts2, ts3 : all trio timestamps (eg, IGs, or anything else)
     tau           : timescale over which we compute the matches
     causal        : if ==0: only the time-distance matters
                     if ==1: only search for x->y "directional" matches
     do_clean      : if ==1: timestamps distant by less than \tau are removed
                     if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
     do_fraction   : if ==0: returns the numbers
                     if ==1: returns the fractions
    
     returned      : the total number of duos found, 
                     or the fraction (divided by 6(=3!) (cf bidirectional==1))
    '''
    do_sanity_checks = 0
    bidirectional = 1 # let's keep this possibility, although it is now deprecated (should alwways be =1)
    # first, we just rename the variables, and call them IGs:
    IG1=ts1 # not clean (yet)
    IG2=ts2
    IG3=ts3

    if (do_clean==1):
        IG1 = clean_IG(IG1, tau=tau)
        IG2 = clean_IG(IG2, tau=tau)
        IG3 = clean_IG(IG3, tau=tau)
        
    nb_IG1 = IG1.shape[0]
    nb_IG2 = IG2.shape[0]
    nb_IG3 = IG3.shape[0]
    
    # then search for duos of IGs:
    nb_duo_12 = find_IG_duo(IG1, IG2, tau=tau, causal=causal, do_clean=do_clean).shape[0]
    nb_duo_13 = find_IG_duo(IG1, IG3, tau=tau, causal=causal, do_clean=do_clean).shape[0]
    nb_duo_23 = find_IG_duo(IG2, IG3, tau=tau, causal=causal, do_clean=do_clean).shape[0]
        
    if (bidirectional==1): # default since 2022-03-30, should not be de-activated
        nb_duo_21 = find_IG_duo(IG2, IG1, tau=tau, causal=causal, do_clean=do_clean).shape[0]
        nb_duo_31 = find_IG_duo(IG3, IG1, tau=tau, causal=causal, do_clean=do_clean).shape[0]
        nb_duo_32 = find_IG_duo(IG3, IG2, tau=tau, causal=causal, do_clean=do_clean).shape[0]
#        if (nb_duo_12!=nb_duo_21): 
#            print("1-2 not symetrical, tau =", tau)
#            print("\t IG1", IG1)
#            print("\t IG2", IG2)
#            print("\t 1-2", find_IG_duo(IG1, IG2, tau=tau, do_clean=do_clean))
#            print("\t 2-1", find_IG_duo(IG2, IG1, tau=tau, do_clean=do_clean))
#        if (nb_duo_13!=nb_duo_31): print("1-3 not symetrical")
#        if (nb_duo_23!=nb_duo_32): print("2-3 not symetrical")
    else:
        nb_duo_21 = 0
        nb_duo_31 = 0
        nb_duo_32 = 0
#    print("1-2 : %d, 1-3 : %d, 2-3 : %d" %(nb_duo_12, nb_duo_13, nb_duo_23))
#    print("2-1 : %d, 3-1 : %d, 3-2 : %d" %(nb_duo_21, nb_duo_31, nb_duo_32))

# sanity checks (only for debug, now useless):
    if (do_sanity_checks>0):
        if (nb_duo_12>nb_IG2):
            print("more duets 1-2", find_IG_duo(IG1, IG2, tau=tau, causal=causal, do_clean=do_clean), "than IGs! ", IG2)
        if (nb_duo_21>nb_IG1):
            print("more duets 2-1", find_IG_duo(IG2, IG1, tau=tau, causal=causal, do_clean=do_clean), "than IGs! ", IG1)
        if (nb_duo_13>nb_IG3):
            print("more duets 1-3 (%d)" %nb_duo_13, find_IG_duo(IG1, IG3, tau=tau, causal=causal, do_clean=do_clean), "than IGs (%d)! "%nb_IG3, IG3)
        if (nb_duo_31>nb_IG1):
            print("more duets 3-1", find_IG_duo(IG3, IG1, tau=tau, causal=causal, do_clean=do_clean), "than IGs! ", IG1)
        if (nb_duo_23>nb_IG3):
            print("more duets 2-3", find_IG_duo(IG2, IG3, tau=tau, causal=causal, do_clean=do_clean), "than IGs! ", IG3)
        if (nb_duo_32>nb_IG2):
            print("more duets 3-2", find_IG_duo(IG3, IG2, tau=tau, causal=causal, do_clean=do_clean), "than IGs! ", IG2)
        
    if (do_fraction==1):
        if (nb_IG1>0):
            nb_duo_21 /= nb_IG1
            nb_duo_31 /= nb_IG1
        if (nb_IG2>0):
            nb_duo_32 /= nb_IG2
            nb_duo_12 /= nb_IG2
        if (nb_IG3>0):
            nb_duo_13 /= nb_IG3
            nb_duo_23 /= nb_IG3

    result = nb_duo_12+nb_duo_13+nb_duo_23+nb_duo_21+nb_duo_31+nb_duo_32
    if (bidirectional==1): result = result/2  
    if (do_fraction==1):   result = result/3
            
    return(result)



# function to count the (total) number of duets/duos as "simultaneous" IGs in a trio
#
# data          : all trio information (from sliders)
# threshold     : minimal variation in the slider signal that ensures there is an IG
# tau           : timescale over which we compute the matches
# bidirectional : if set to 1, then look at x->y and y->x     # deprecated 2022-03-30
# causal        : if ==0: then look at causal (or directional) interactions
# do_clean      : if ==1: timestamps distant by less than \tau are removed
#                 if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
# do_fraction   : if ==0: returns the numbers
#                 if ==1: returns the fractions
#
# returned : the total number of duos found (so must be divided by 2 if bi-directional)
def count_nb_duo(data, threshold=4, tau=2, causal=0, do_clean=1, do_fraction=0):
    ''' function to count the (total) number of "simultaneous duets" IGs in a trio
     data          : the trio data (expected: slider data)
     tau           : timescale over which we compute the matches
     causal        : if ==0: only the time-distance matters
                     if ==1: only search for x->y "directional" matches
     do_clean      : if ==1: timestamps distant by less than \tau are removed
                     if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
     do_fraction   : if ==0: returns the numbers
                     if ==1: returns the fractions
    
     returned      : the total number of duos found, 
                     or the fraction (divided by 6(=3!) (cf bidirectional==1))
     '''
    
    # first, find "individual" IGs:
    IG1=find_IG(data[0,:,1], threshold=threshold) # not clean (yet)
    IG2=find_IG(data[1,:,1], threshold=threshold)
    IG3=find_IG(data[2,:,1], threshold=threshold)

    # then duos of IGs:
    return(count_duo_matches_in_trio(IG1, IG2, IG3, tau=tau, 
                                     causal=causal, do_clean=do_clean, do_fraction=do_fraction))



# function to count the (total) number of triplets as"simultaneous" timestamps (eg, IGs) in a trio
#
# ts1, ts2, ts3 : all trio timestamps (eg, IGs, or anything else)
# tau           : timescale over which we compute the matches
# bidirectional : if set to 1, then look at x->y and y->x       # deprecated 2022-03-30
# causal        : if ==0: then look at causal (or directional) interactions
# do_clean      : if ==1: timestamps distant by less than \tau are removed
#                 if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
# do_fraction   : if ==0: returns the numbers
#                 if ==1: returns the fractions
#
# returned : the total number of duos found, so
# - divided by 2(=2!) if bi-directional==1 (default)
# - divided by 6(=3!) if output is such that do_fraction==1
def count_trio_matches_in_trio(ts1, ts2, ts3, tau=2, causal=0, do_clean=1, do_fraction=0):
    ''' function to count the (total) number of "simultaneous" timestamps (eg, IGs) in a trio
     -> this function is for "triplets"
      
     ts1, ts2, ts3 : all trio timestamps (eg, IGs, or anything else)
     tau           : timescale over which we compute the matches
     causal        : if ==0: only the time-distance matters
                     if ==1: only search for x->y "directional" matches
     do_clean      : if ==1: timestamps distant by less than \tau are removed
                     if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
     do_fraction   : if ==0: returns the numbers
                     if ==1: returns the fractions
    
     returned      : the total number of duos found, 
                     or the fraction (divided by 6(=3!) (cf bidirectional==1))
    '''
    do_sanity_checks = 0
    bidirectional = 1 # let's keep this possibility, although it is now deprecated (should alwways be =1)
    # first, we just rename the variables, and call them IGs:
    IG1=ts1 # not clean (yet)
    IG2=ts2
    IG3=ts3
    
    if (do_clean==1):
        IG1 = clean_IG(IG1, tau=tau)
        IG2 = clean_IG(IG2, tau=tau)
        IG3 = clean_IG(IG3, tau=tau)
        
    nb_IG1 = IG1.shape[0]
    nb_IG2 = IG2.shape[0]
    nb_IG3 = IG3.shape[0]

    # then search for (bidirectional) duets of IGs:
    duo_12 = find_IG_duo(IG1, IG2, tau=tau, causal=causal, do_clean=do_clean)
    duo_13 = find_IG_duo(IG1, IG3, tau=tau, causal=causal, do_clean=do_clean)
    duo_23 = find_IG_duo(IG2, IG3, tau=tau, causal=causal, do_clean=do_clean)
    duo_12 = numpy.append(duo_12, find_IG_duo(IG2, IG1, tau=tau, causal=causal, do_clean=do_clean))
    duo_13 = numpy.append(duo_13, find_IG_duo(IG3, IG1, tau=tau, causal=causal, do_clean=do_clean))
    duo_23 = numpy.append(duo_23, find_IG_duo(IG3, IG2, tau=tau, causal=causal, do_clean=do_clean))

    # then search for (bidirectional) triplets:
    trio_1 = find_IG_duo(duo_12, duo_13, tau=tau, causal=causal, do_clean=do_clean)
    trio_2 = find_IG_duo(duo_12, duo_23, tau=tau, causal=causal, do_clean=do_clean)
    trio_3 = find_IG_duo(duo_23, duo_13, tau=tau, causal=causal, do_clean=do_clean)
    trio_1 = numpy.append(trio_1, find_IG_duo(duo_13, duo_12, tau=tau, causal=causal, do_clean=do_clean))
    trio_2 = numpy.append(trio_2, find_IG_duo(duo_23, duo_12, tau=tau, causal=causal, do_clean=do_clean))
    trio_3 = numpy.append(trio_3, find_IG_duo(duo_13, duo_23, tau=tau, causal=causal, do_clean=do_clean))
        
    nb_trio_1 = trio_1.shape[0]/2 # division by 2 because bi-directional
    nb_trio_2 = trio_2.shape[0]/2
    nb_trio_3 = trio_3.shape[0]/2
    
    if (do_fraction==1):
        if (nb_IG1>0): nb_trio_1 /= nb_IG1
        if (nb_IG2>0): nb_trio_2 /= nb_IG2
        if (nb_IG3>0): nb_trio_3 /= nb_IG3
        
    result = nb_trio_1 + nb_trio_2 + nb_trio_3
    if (do_fraction==1): result /= 3
            
    return(result)
    
  
# function to count the (total) number of triplets/triets as "simultaneous" IGs in a trio
#
# data          : all trio information (from sliders)
# threshold     : minimal variation in the slider signal that ensures there is an IG
# tau           : timescale over which we compute the matches
# causal        : if ==0: then look at causal (or directional) interactions
# do_clean      : if ==1: timestamps distant by less than \tau are removed
#                 if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
# do_fraction   : if ==0: returns the numbers
#                 if ==1: returns the fractions
#
# returned : the total number of duos found (so must be divided by 2 if bi-directional)
def count_nb_trio(data, threshold=4, tau=2, causal=0, do_clean=1, do_fraction=0):
    ''' function to count the (total) number of "simultaneous triets" IGs in a trio
     data          : the trio data (expected: slider data)
     tau           : timescale over which we compute the matches
     causal        : if ==0: only the time-distance matters
                     if ==1: only search for x->y "directional" matches
     do_clean      : if ==1: timestamps distant by less than \tau are removed
                     if ==0: only redondant timestamps are removed (equivalent to clean_IG with tau=0)
     do_fraction   : if ==0: returns the numbers
                     if ==1: returns the fractions
    
     returned      : the total number of duos found, or the fraction
     '''
    
    # first, find "individual" IGs:
    IG1=find_IG(data[0,:,1], threshold=threshold) # not clean (yet)
    IG2=find_IG(data[1,:,1], threshold=threshold)
    IG3=find_IG(data[2,:,1], threshold=threshold)

    # then duos of IGs:
    return(count_trio_matches_in_trio(IG1, IG2, IG3, tau=tau, 
                                     causal=causal, do_clean=do_clean, do_fraction=do_fraction))

 

if __name__ == "__main__":
    print("use this module by importing it in your Python script or Jupyter notebook")
    print("e.g., with the following line:")
    print("import impro")