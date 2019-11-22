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
from searchAgents import mazeDistance
import random, util
import math

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsulePositions = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"
    

        " Distance to closest ghost "
        closestGhost = min(newGhostStates, key=lambda g: manhattanDistance(newPos, g.getPosition()))
        closestGhostDistance = manhattanDistance(newPos, closestGhost.getPosition())

        if(closestGhostDistance <= 1):
            return -10000

        foodList = newFood.asList()
        numFood = len(foodList)

        totalFoodDistance = sum(manhattanDistance(newPos, item) for item in foodList)

        if(newPos in capsulePositions):
            return float('Inf')

        if(numFood == 0):
            return float('Inf')

        score = .5*closestGhostDistance - (50*numFood) - totalFoodDistance/numFood 

        return score


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, agentNum, depth):
            if depth == 0:
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agentNum)
            successorValueList = map(lambda action: value(state.generateSuccessor(agentNum, action), agentNum, depth), legalActions)
            v = max(successorValueList)
            return v

        def minValue(state, agentNum, depth):
            legalActions = state.getLegalActions(agentNum)
            successorValueList = map(lambda action: value(state.generateSuccessor(agentNum, action), agentNum, depth), legalActions)
            v = min(successorValueList)
            return v

        def value(state, agentNum, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentNum == state.getNumAgents() - 1:
                agent = 0
            else:
                agent = agentNum + 1

            if agent == 0:
                return maxValue(state, agent, depth - 1)
            else:
                return minValue(state, agent, depth)


        agent = 0
        legalActions = gameState.getLegalActions()
        return max(legalActions, key=lambda a: value(gameState.generateSuccessor(0, a), agent, self.depth))
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, agentNum, depth, alpha, beta):
            if depth == 0:
                return (self.evaluationFunction(state), None)
            legalActions = state.getLegalActions(agentNum)
            v = (float("-inf"), None)
            for action in legalActions:
                successor = state.generateSuccessor(agentNum, action)
                val = value(successor, agentNum, depth, alpha, beta)
                if v[0] < val[0]:
                    v = (val[0], action)
                    
                if v[0] > beta:
                    return (v[0], action)
                alpha = max(alpha, v[0])
            return v

        def minValue(state, agentNum, depth, alpha, beta):
            legalActions = state.getLegalActions(agentNum)
            v = (float("+inf"), None)
            for action in legalActions:
                successor = state.generateSuccessor(agentNum, action)
                val = value(successor, agentNum, depth, alpha, beta)
                if v[0] > val[0]:
                    v = (val[0], action)
                    
                if v[0] < alpha:
                    return (v[0], action)
                beta = min(beta, v[0])
            return v

        def value(state, agentNum, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), None)

            if agentNum == state.getNumAgents() - 1:
                agent = 0
            else:
                agent = agentNum + 1

            if agent == 0:
                return maxValue(state, agent, depth - 1, alpha, beta)
            else:
                return minValue(state, agent, depth, alpha, beta)

        agent = 0
        legalActions = gameState.getLegalActions()
        
        v = value(gameState, -1, self.depth+1, float("-inf"), float("inf"))
        return v[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, agentNum, depth):
            if depth == 0:
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agentNum)
            successorValueList = map(lambda action: value(state.generateSuccessor(agentNum, action), agentNum, depth), legalActions)
            v = max(successorValueList)
            return v

        def expValue(state, agentNum, depth):
            legalActions = state.getLegalActions(agentNum)
            successorValueList = map(lambda action: value(state.generateSuccessor(agentNum, action), agentNum, depth), legalActions)
            v = sum(successorValueList) / len(legalActions)
            return v

        def value(state, agentNum, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentNum == state.getNumAgents() - 1:
                agent = 0
            else:
                agent = agentNum + 1

            if agent == 0:
                return maxValue(state, agent, depth - 1)
            else:
                return expValue(state, agent, depth)

        agent = 0
        legalActions = gameState.getLegalActions()
        return max(legalActions, key=lambda a: value(gameState.generateSuccessor(0,a), agent, self.depth))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsulePositions = currentGameState.getCapsules()


    curScore = currentGameState.getScore()

    if capsulePositions:
        curScore += 1. / len(capsulePositions)

    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float('-inf')


    scaredGhosts = []
    for ghost in ghostStates:
        if ghost.scaredTimer:
            scaredGhosts.append(ghost)

    closestScared = float("inf")
    if scaredGhosts:
        closestScared = min([mazeDistance(curPos, ghost.getPosition(), currentGameState) for ghost in scaredGhosts])

    if closestScared == 0:
        closestScared = 0.1

    closestGhostDistance = 0
    if ghostStates:
        closestGhostDistance = min([mazeDistance(curPos, ghost.getPosition(), currentGameState) for ghost in ghostStates])
        if closestGhostDistance <= 4:
            curScore += 1.*closestGhostDistance
        else:
            curScore += 4

    foodList = food.asList()
    numFood = len(foodList)

    closestFoodDistance = 0
    if foodList:
        closestFoodDistance = min([mazeDistance(curPos, item,  currentGameState) for item in foodList])

    if closestFoodDistance == 0:
        return float('inf')

    score = curScore \
            + 1./numFood \
            + 1.0/closestFoodDistance \
            + 10./closestScared
    return score

better = betterEvaluationFunction
