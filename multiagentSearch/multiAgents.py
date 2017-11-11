# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, copy

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Initialize
        scoreBro = 0

        # This section creates a value for how much food is left on the field
        numFoodLeft = 0
        for food in newFood:
            if food == True:
                numFoodLeft += 1

        # This section creates a value for how many power pellets are left on the field
        numPowerPelLeft = 0
        for powerP in successorGameState.getCapsules():
            numPowerPelLeft += 1

        # This calculates the distance to the nearest ghost
        distToNearGhost = 100000
        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(ghost, newPos) < distToNearGhost:
                distToNearGhost = manhattanDistance(ghost, newPos)

        # This increments and decrements score bro
        scoreBro += successorGameState.getScore() * 2   # Increment, we want higher score
        scoreBro -= numFoodLeft * 40    # Decrement, we want fewer food pellets
        scoreBro -= numPowerPelLeft * 50    # Decrement we want fewer power pellets

        # Penalize doing nothing
        if action == Directions.STOP or action == Directions.REVERSE:
            scoreBro -= 100

        # HEAVY penalize getting too close to ghost
        if distToNearGhost <= 1:
            scoreBro -= 10000

        # If ghost is scared, go after ghost by minimizing distance to it
        for sTime in newScaredTimes:
            if sTime > 0:
                scoreBro -= distToNearGhost

        # Calculations complete, return final computed score
        return scoreBro


def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # Implementation of value function from pseudo code
    def value(self, gState, agentIndex, depth):

        # If we just finished the turn of the last agent in the ply,
        # go back to first agent and go up a layer of depth
        if agentIndex == gState.getNumAgents():
            depth = depth - 1
            agentIndex = 0

        # If we lost, won, or the depth is 0, no need to do more calculations
        if gState.isWin() or gState.isLose() or depth <= 0:
            return self.evaluationFunction(gState)

        # If agent is pacman, maximize
        if agentIndex == 0:
            return self.maxValue(gState, agentIndex, depth)

        # If agent is a ghost, minimize
        if agentIndex > 0:
            return self.minValue(gState, agentIndex, depth)

    # Maximizer function
    def maxValue(self, gState, agentIndex, depth):
        # Init v to a very small number
        v = -1000000

        # Get the best value out of all possible actions
        for action in gState.getLegalActions(agentIndex):

            # Ignore the stop action
            if action != Directions.STOP:

                # Create a new successor out of the action we are looking at
                newSucc = gState.generateSuccessor(agentIndex, action)

                # Recursively call the value function on the next agent
                # Eventually a value will bubble up through all the function calls
                v = max(v, self.value(newSucc, agentIndex + 1, depth))

        # Return best value
        return v

    # Minimizer function
    def minValue(self, gState, agentIndex, depth):
        # Init v to a very large number
        v = 1000000

        # Get the best value out of all possible actions
        for action in gState.getLegalActions(agentIndex):

            # Ignore the stop action
            if action != Directions.STOP:

                # Create a new successor out of the action we are looking at
                newSucc = gState.generateSuccessor(agentIndex, action)

                # Recursively call the value function on the next agent
                # Eventually a value will bubble up through all the function calls
                v = min(v, self.value(newSucc, agentIndex + 1, depth))

        # Return best value
        return v

    # Returns an action for pacman (maximizer) based on the values returned by the value function
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Init score to a low value
        maxScore = -1000000

        # Init what will be the best action
        bestAction = "NA"

        # For each legal action, except for STOP, use value function to find the best action
        for action in gameState.getLegalActions():
            if action != Directions.STOP:
                successor = gameState.generateSuccessor(0, action)
                score = self.value(successor, 1, self.depth)

                # If current score is better than the max score, update best score and best action
                if score > maxScore:
                    maxScore = score
                    bestAction = action

        # Then return the best action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # Implementation of value function from pseudo code
    def value(self, gState, agentIndex, depth, alpha, beta):

        # If we just finished the turn of the last agent in the ply,
        # go back to first agent and go up a layer of depth
        if agentIndex == gState.getNumAgents():
            depth = depth - 1
            agentIndex = 0

        # If we lost, won, or the depth is 0, no need to do more calculations
        if gState.isWin() or gState.isLose() or depth <= 0:
            return self.evaluationFunction(gState)

        # If agent is pacman, maximize
        if agentIndex == 0:
            return self.maxValue(gState, agentIndex, depth, alpha, beta)

        # If agent is ghost, minimize
        if agentIndex > 0:
            return self.minValue(gState, agentIndex, depth, alpha, beta)

    # Maximizer function
    def maxValue(self, gState, agentIndex, depth, alpha, beta):

        # Init value to very small
        v = -100000

        # Get the best value out of all possible actions except for STOP (because we hate stop)
        for action in gState.getLegalActions(agentIndex):
            if action != Directions.STOP:
                newSucc = gState.generateSuccessor(agentIndex, action)

                # Obtain the max of the value and a recursive call of value on the NEXT state
                v = max(v, self.value(newSucc, agentIndex + 1, depth, alpha, beta))

                # If v is greater than or equal to beta, prune
                if v >= beta:
                    return v

                # Update alpha
                alpha = max(alpha, v)

        # Return best value
        return v

    # Minimizer function
    def minValue(self, gState, agentIndex, depth, alpha, beta):

        # Init value to very large
        v = 100000

        # Get the best value out of all possible actions except for STOP (because we hate stop)
        for action in gState.getLegalActions(agentIndex):
            if action != Directions.STOP:
                newSucc = gState.generateSuccessor(agentIndex, action)

                # Obtain the min of the value and a recursive call of value on the NEXT state
                v = min(v, self.value(newSucc, agentIndex + 1, depth, alpha, beta))

                # if v is less than or equal to alpha, prune
                if v <= alpha:
                    return v

                # Update beta
                beta = min(beta, v)

        # Return best value
        return v

    # Initializes alpha and beta then calls value function until it finds best action, then returns best action
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = -100000000
        beta = 100000000
        score = -1000000
        maxScore = -1000000
        bestAction = "NA"
        for action in gameState.getLegalActions():
            if action != Directions.STOP:
                successor = gameState.generateSuccessor(0, action)
                score = self.value(successor, 1, self.depth, alpha, beta)
                if score > maxScore:
                    maxScore = score
                    bestAction = action
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Implementation of value function from pseudo code
    def value(self, gState, agentIndex, depth):

        # If we just finished the turn of the last agent in the ply,
        # go back to first agent and go up a layer of depth
        if agentIndex == gState.getNumAgents():
            depth = depth - 1
            agentIndex = 0

        # If we lost, won, or the depth is 0, no need to do more calculations
        if gState.isWin() or gState.isLose() or depth <= 0:
            return self.evaluationFunction(gState)

        # If pacman turn, maximize
        if agentIndex == 0:
            return self.maxValue(gState, agentIndex, depth)

        # If ghost turn, calculate expected value
        if agentIndex > 0:
            return self.expValue(gState, agentIndex, depth)

    # Maximizer function
    def maxValue(self, gState, agentIndex, depth):

        # Init value to very low
        v = -100000

        # Get best score from all the actions (avoid STOP because we hate it)
        for action in gState.getLegalActions(agentIndex):
            if action != Directions.STOP:
                newSucc = gState.generateSuccessor(agentIndex, action)

                # Get max between this value and the next state value
                v = max(v, self.value(newSucc, agentIndex + 1, depth))

        # Return best value
        return v

    # Expected value function
    def expValue(self, gState, agentIndex, depth):
        # Init value to zero
        v = 0

        # Probability is equal for all actions, so divide 1.0 by number of legal actions
        p = 1.0 / len(gState.getLegalActions(agentIndex))
        for action in gState.getLegalActions(agentIndex):
            if action != Directions.STOP:
                newSucc = gState.generateSuccessor(agentIndex, action)
                v += p * self.value(newSucc, agentIndex + 1, depth)
        return v

    # Chooses best action based on score computed via calling value function
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        score = -1000000
        maxScore = -1000000
        bestAction = 'NA'
        for action in gameState.getLegalActions():
            if action != Directions.STOP:
                successor = gameState.generateSuccessor(0, action)
                score = self.value(successor, 1, self.depth)
                if score > maxScore:
                    maxScore = score
                    bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Initialize value counter for the state
    scoreBro = 0

    # Calculates number of food left
    numFoodLeft = 0
    for food in newFood:
        if food == True:
            numFoodLeft += 1

    # Calculates number of power pellets left
    numPowerPelLeft = 0
    for powerP in currentGameState.getCapsules():
        numPowerPelLeft += 1

    # Calculate distance to nearest ghost
    distToNearGhost = 100000
    for ghost in currentGameState.getGhostPositions():
        if manhattanDistance(ghost, newPos) < distToNearGhost:
            distToNearGhost = manhattanDistance(ghost, newPos)

    # Calculate distance to nearest food
    distToClosestFood = 100000
    for food in currentGameState.getFood():
        if food == True:
            if manhattanDistance(food, newPos) < distToClosestFood:
                distToClosestFood = manhattanDistance(food, newPos)

    # The if statements are to eliminate the posibility of by-zero division
    scoreBro += currentGameState.getScore()     # Increment value of state by the current game score
    if numFoodLeft != 0:
        scoreBro += 1 / numFoodLeft * 50    # Minimize amount of food left
    if numPowerPelLeft != 0:
        scoreBro += 1 / numPowerPelLeft * 20    # Minimize amount of power pellets left
    if distToClosestFood != 0:
        scoreBro += 1 / distToClosestFood * 50  # Minimize distance to closest food
    if distToNearGhost != 0:
        scoreBro += 1 / distToNearGhost     # Maximize distance to nearest ghost

    # If get too close to ghost, big penalty
    if distToNearGhost <= 1:
        scoreBro += 1 / 10000

    for sTime in newScaredTimes:
        if sTime > 0:
            scoreBro += 1 / distToNearGhost * 2     # If ghost is scared, add reward for going after ghost

    return scoreBro     # Return final calculated value

# Abbreviation
better = betterEvaluationFunction