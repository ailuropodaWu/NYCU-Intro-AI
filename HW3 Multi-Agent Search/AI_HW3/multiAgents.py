from util import manhattanDistance
from game import Directions
import random, util
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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """ 
        minimax function:
            Used to do the minimax search.
            The Pacman is the max-level and the ghosts are all the mini-level.
            The value is calculated by the evaluation function with the state when reaching the bottom.
            Use list() ValNodes to record all subnodes' value after the current state doing different legal actions.

            Parameters:
                state:
                    The gameState after the action done based on the prior state
                agentIndex:
                    The agent index used to determine which level is, and also decide who need to action to change the state.
                depth:
                    The current depth of the minimax search tree, if depth was equal to the self.depth, it means reaching the bottom
            Return:
                Return the final actions when the depth is 1, otherwise, return the min or the max value.
        
        Return:
            Minimax search beginning with the depth 1 and the agent Pacman
        """
        
        agentNum = gameState.getNumAgents()
        
        def minimax(state, agentIndex, depth):
            if depth > self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            valNodes = []
            legalActions = state.getLegalActions(agentIndex)
            
            if 'Stop' in legalActions:
                legalActions.remove('Stop')
            
            if agentIndex == 0:
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = minimax(successor, 1, depth)
                    valNodes.append(val)
                if depth == 1:
                    for i in range(len(valNodes)):
                        if valNodes[i] == max(valNodes):
                            return legalActions[i]
                else:
                    return max(valNodes)
            else:
                nextAgent = agentIndex + 1
                if agentIndex == agentNum - 1:
                    nextAgent = 0
                    depth += 1
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = minimax(successor, nextAgent, depth)
                    valNodes.append(val)
                return min(valNodes)
                
        return minimax(gameState, 0, 1)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """ 
        minimax function:
            Based on the minimax function in part1, but having parameters alpha and beta to optimize the searching speed.

            Parameters:
                state, agentIndex, depth:
                    As same as those in part1.
                alpha:
                    Used to determine whether the mini-level could break, and it is maintained by the max-level.
                beta:
                    Similar to alpha, but maintained by mini-level, and used to break the max-level.
            Return:
                Having the same return as part1.
        
        Return:
            Having the same beginning agentIndex and depth, and initializing alpha to negative infinity, beta to the positive one.
        """
        agentNum = gameState.getNumAgents()
        
        def minimax(state, agentIndex, depth, alpha, beta):
            if depth > self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            valNodes = []
            legalActions = state.getLegalActions(agentIndex)
            
            if 'Stop' in legalActions:
                legalActions.remove('Stop')
            
            if agentIndex == 0:
                ret = float("-inf")
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = minimax(successor, 1, depth, alpha, beta)
                    valNodes.append(val)
                    alpha = max(alpha, max(valNodes))
                    ret = max(valNodes)
                    if ret > beta:
                        break
                if depth == 1:
                    for i in range(len(valNodes)):
                        if valNodes[i] == max(valNodes):
                            return legalActions[i]
                else:
                    return ret
            else:
                nextAgent = agentIndex + 1
                if agentIndex == agentNum - 1:
                    nextAgent = 0
                    depth += 1
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = minimax(successor, nextAgent, depth, alpha, beta)
                    valNodes.append(val)
                    beta = min(beta, min(valNodes))
                    if min(valNodes) < alpha:
                        return min(valNodes)
                return min(valNodes)
        
        return minimax(gameState, 0, 1, float("-inf"), float("inf"))
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """ 
        expectimax function:
            Having the same structure as minimax, but the mini-level has changed to the expectation-level.

            Parameters:
                All parameters have the same usage as part1
            Return:
                Using the same return way as minimax, but change the it to expection instead of minimum.
        
        Return:
            Expectimax search with the same beginning parameters as part1.
        """
        def expectimax(state, agentIndex, depth):
            agentNum = gameState.getNumAgents()

            if depth > self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            valNodes = []
            legalActions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = expectimax(successor, 1, depth)
                    valNodes.append(val)
                if depth == 1:
                    for i in range(len(valNodes)):
                        if valNodes[i] == max(valNodes):
                            return legalActions[i]
                else:
                    return max(valNodes)
            else:
                nextAgent = agentIndex + 1
                if agentIndex == agentNum - 1:
                    nextAgent = 0
                    depth += 1
                for action in legalActions:
                    successor = state.getNextState(agentIndex, action)
                    val = expectimax(successor, nextAgent, depth)
                    valNodes.append(val)
                return sum(valNodes) / len(valNodes)
        return expectimax(gameState, 0, 1)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """ 
    Variables:
        currentScore: The score of current state
        currentPos: The position of the Pacman
        ghostStates: The states of all ghosts
        currentFood: The distance from currentPos to all foods
        currentCapsules: The distance from currentPos to all capsules
    
    Return:
        Find the minimum ghost and scared ghost distance, and the nearest food and capsule.
        And given 4 conditions to have different strategy.
        1. If there is a scared ghost: Go to find the closet one.
        2. If the ghost is very close: Keep the life and run away.
        3. If there are still some capsules: Go to find the capsule through the way with more food.
        4. General case: Balance between eating the foods and keeping the life.
    """
    currentScore = currentGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    currentFood = [manhattanDistance(currentPos, food) for food in currentGameState.getFood().asList()]
    currentCapsules = [manhattanDistance(currentPos, capsule) for capsule in currentGameState.getCapsules()]

    minGhostDistance = float("inf")
    minScaredGhostDistance = float("inf")
    for state in ghostStates:
        distance = manhattanDistance(currentPos, state.getPosition())
        minGhostDistance = min(distance, minGhostDistance)
        if state.scaredTimer > 0:
            minScaredGhostDistance = min(distance, minScaredGhostDistance)

    nearestFood = 1 if len(currentFood) == 0 else min(currentFood)
    nearestCapsules = 0 if len(currentCapsules) == 0 else min(currentCapsules)

    if minScaredGhostDistance < float("inf"):
        return currentScore - minScaredGhostDistance
    elif minGhostDistance < 2:
        return currentScore + minGhostDistance
    elif len(currentCapsules) > 0:
        return currentScore - 2 * nearestCapsules - nearestFood
    return currentScore - nearestFood + minGhostDistance

    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
