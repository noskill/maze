import logging
import Queue


logger = logging.getLogger(__name__)


def search(initial, is_goal, heuristic, actions):
    """
    Searches for path from initial state to goal state using A* algorithm
    Parameters
    -------
    initial: tuple
        ((x,y), 3x3 orientation)
    is_goal: callable
        accepts state, returns true if state is a goal state
    heuristic: callable
        accepts state, returns heuristic distance to a goal state
    actions: callable
        accepts current state, returns (new_state, action leading to new_state) pairs

    Returns
    -------
    list of (action to apply in state, state) pairs leading from initial state to a goal state
    """
    # path_cost={curretn_pos:prev_pos,cost      , prev action}
    path_cost = { initial: (None, heuristic(initial), None)}
    open = set()
    open.add(initial)
    explored = set()
    queue = Queue.PriorityQueue()
    #todo remove state access
    import pdb;pdb.set_trace()
    queue.put((path_cost[initial][1], initial))

    found = False  # flag that is set when search is complet
    resign = False # flag set if we can't find expand

    while not found and not resign:
        if queue.qsize() == 0:
            resign = True
            continue

        min_f, min_pos = queue.get()
        if min_pos in open:
            open.remove(min_pos)
        logger.info("Exploring: ", min_pos, "cost: ", min_f)
        if is_goal(min_pos):
            found = True
            goal = min_pos
            break
        min_pos_cost = min_f - heuristic(min_pos)
        for (new_pos, action) in actions(min_pos):
            new_cost = min_pos_cost + 1 + heuristic(new_pos)
            #  if new pos not in open or in open with higher cost
            if new_pos not in explored and (new_pos not in open or new_cost < path_cost[new_pos][1]):
                open.add(new_pos)
                logger.info("Adding: prev pos ", min_pos,"action: ", action, "new_pos ",  new_pos, "cost ",  new_cost)
                path_cost[new_pos] = (min_pos, new_cost, action)
                queue.put((new_cost, new_pos))
        #place node in explored
        explored.add(min_pos)
    result = []
    if found:
        prev_pos = path_cost[goal][0]
        prev_act = path_cost[goal][2]
        result.append((prev_act, prev_pos))
        logger.info("appending to result ", prev_pos, prev_act)
        while prev_pos is not None and result[-1][1] != initial:
            prev_pos = path_cost[result[-1][1]][0]
            prev_act = path_cost[result[-1][1]][2]
            logger.info("appending to result ", prev_pos, prev_act)
            result.append((prev_act, prev_pos))
    return result

