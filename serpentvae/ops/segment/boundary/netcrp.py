"""
This file contains the implementation of the non-parametric clustering model that we use to cluster concept tokens.

It is based on the network chinese restaurant process (NetCRP), 
which allows the use of a graph to constrain how data is clustered

We view the sequnce of sequence tokens as a graph reminiscent of a singly linked list, 
and have NetCRP figure out where to remove edges,
forming the contiguous clusters


"""