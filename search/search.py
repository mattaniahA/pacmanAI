""" 
Mattaniah Aytenfsu & Elyse Shackleton
CS 430 
Assignment: Project 1
Due Date: 10/1/2017
"""
import new


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    *** YOUR CODE HERE ***"""
    
    fringe = util.Stack() # Making a new queue called fringe -- This is our queuing strategy
    explored = set() # Declare a set to contain explored nodes
    path = [] # Path placeholder

    # Create start state and push onto fringe
    location = problem.getStartState()[0]
    corners = problem.getStartState()[1]
    #Dr. Dan - And same here...state and corners needs to be a Tuple, not a list
    #fringe.push([location, corners, path, 0])
    fringe.push((location, tuple(corners), path, 0))

    
    # Main while-loop; peels item off the fringe, and adds visited node's successors to the fringe
    while(fringe.isEmpty() != True): 
        currState = fringe.pop()
        
        # Dr. Dan - Notice the error on the output: says explored is "unhashable type: 'list'
        # this needs to be a tuple, not a list
        #simpleState = [copy.copy(currState[0]), copy.copy(currState[1])] # Use simple state representation to keep track of explored states
        simpleState = (copy.copy(currState[0]), copy.copy(currState[1])) # Use simple state representation to keep track of explored states
        
        # If current state has not yet been visited since the last corner, continue
        if (simpleState not in explored):
           
            #Add current state to list of explored coordinates
            explored.add(simpleState)
            
            # Check current state to see if it's a goal state
            # Dr. Dan - You should be checking simpleState, not currState
            #if(problem.isGoalState(currState)):
            if(problem.isGoalState(simpleState)):
                return currState[2] # If goal state, return path            
            
            # Iterate through current state's successors and push them all onto the fringe
            for child in problem.getSuccessors(currState):
                # Create a new child state, and push that onto the fringe
                # Dr. Dan - And same here - Python did not like this being a list...wanted a tuple

                newChildState = (child[0],  child[1], child[2],      child[3] + currState[3])
                
                fringe.push(newChildState)
            

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue() # Making a new queue called fringe -- This is our queuing strategy
    explored = set() # Declare a set to contain explored nodes
    path = [] # Path placeholder

    # Create start state and push onto fringe
    startState = [problem.getStartState(), path, 0]

    #Dr. Dan - And same here...state and corners needs to be a Tuple, not a list
    fringe.push(startState)

    
    # Main while-loop; peels item off the fringe, and adds visited node's successors to the fringe
    while(fringe.isEmpty() != True): 
        currState = fringe.pop()
        
        # Dr. Dan - Notice the error on the output: says explored is "unhashable type: 'list'
        # this needs to be a tuple, not a list
        simpleState = (copy.copy(currState[0][0]), copy.copy(currState[0][1])) # Use simple state representation to keep track of explored states
        
        # If current state has not yet been visited since the last corner, continue
        if (simpleState not in explored):
           
            #Add current state to list of explored coordinates
            explored.add(simpleState)
            
            # Check current state to see if it's a goal state
            # Dr. Dan - You should be checking simpleState, not currState
            if(problem.isGoalState(simpleState)):
                return currState[1] # If goal state, return path            
            
            # Iterate through current state's successors and push them all onto the fringe
            for child in problem.getSuccessors(currState):
                # Create a new child state, and push that onto the fringe, child[2] contains path so far, child[3] contains cost
                # Dr. Dan - And same here - Python did not like this being a list...wanted a tuple
                newChildState = ((child[0][0],  child[0][1]), child[1],      child[2] + currState[2])
                
                fringe.push(newChildState)
            
            
    
        

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue() # Make a new priority queue called fringe -- This is our queuing strategy
    explored = set() # Declare a set to contain explored nodes
    path = [] # Create placeholder path that we can use when creating the start state
    
    # Create template for currState that contains coordinates, another list of actions, and sum of total path cost
    currState = ["( , )", path, 0]
    
    # Create start state and push onto fringe
    startState = [problem.getStartState(), path, 0]
    fringe.push(startState, problem.getCostOfActions(startState[1]))
    
    # Main while-loop; peels item off the fringe, and adds visited node's successors to the fringe
    # Loop executes as long as the fringe is not empty
    while(fringe.isEmpty() != True):   
        currState = fringe.pop()
        
        # Check coordinates of current visited node to see if it's a goal state
        if(problem.isGoalState(currState[0])):
            return currState[1] # If goal state, return path to goal state
        
        # Check to see if current state has previously been visited; if not, continue
        if (currState[0] not in explored):
            
            # Add current state to list of explored coordinates
            explored.add(currState[0])         
            
            # Iterate through current state's successors and push them all onto the fringe
            for child in problem.getSuccessors(currState[0]):
                pathToParent = copy.copy(currState[1]) # pathToParent contains the path accumulated by the parent so far
                
                # Create a new child state, containing attributes of parent
                # We add the cost and path accumulated by the parent so far to those of the child
                newChildState = (child[0], pathToParent + [child[1]],      child[2] + currState[2])
                
                # use .getCostOfActions to find the priority of the new child state we created, and push onto the fringe with that priority
                priority = problem.getCostOfActions(newChildState[1])
                fringe.push(newChildState, priority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue() # Making a new queue called fringe -- This is our queuing strategy
    explored = set() # Declare a set to contain explored nodes
    path = [] # Path placeholder

    # Create start state and push onto fringe
    startState = [problem.getStartState(), path, 0]
    #Dr. Dan - And same here...state and corners needs to be a Tuple, not a list
    #fringe.push(location, tuple(corners), path)
    fringe.push(startState, 1) # Priority doesn't matter, because it's the only thing on the fringe
    
    # Main while-loop; peels item off the fringe, and adds visited node's successors to the fringe
    while(fringe.isEmpty() != True): 
        currState = fringe.pop()
        
        # Dr. Dan - Notice the error on the output: says explored is "unhashable type: 'list'
        # this needs to be a tuple, not a list
        #simpleState = (copy.copy(currState[0][0]), copy.copy(currState[0][1]) ) # Use simple state representation to keep track of explored states
        # Dr. Dan 2 - The above line was too specific...just need to grab whatever state is (which can be composed of multiple things
        simpleState = copy.copy(currState[0])

        # If current state has not yet been visited since the last corner, continue
        if (simpleState not in explored):
           
            #Add current state to list of explored coordinates
            explored.add(simpleState)
            
            # Check current state to see if it's a goal state
            # Dr. Dan - You should be checking simpleState, not currState
            if(problem.isGoalState(simpleState)):
                return currState[1] # If goal state, return path            
            
            # Iterate through current state's successors and push them all onto the fringe
            # Dr. Dan - Shouuld just be simpleState getting passed in
            for child in problem.getSuccessors(currState):
                # Create a new child state, and push that onto the fringe, child[2] contains path so far, child[3] contains cost
                # Dr. Dan 2 - Better way to do cost, plus proper state piece needs to be passed into heuristic

                gCost = currState[2] + child[2]
                hCost = heuristic(child[0], problem)
                fCost = gCost + hCost
                path = child[1]
                newChildState = (child[0], path, gCost)
                fringe.push(newChildState, fCost)
                
                #priority = problem.getCostOfActions(child[1]) + heuristic(child, problem)
                # Dr. Dan - And same here - Python did not like this being a list...wanted a tuple
                
                #newChildState = ( (child[0][0], child[0][1]),  child[1],      child[2] + currState[2])
                #fringe.push(newChildState, priority)
            
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch