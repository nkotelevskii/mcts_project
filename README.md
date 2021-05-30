### Evader-pursuer zero-sum game by MCTS

The repository contains the code of the final project "Evader-pursuer zero-sum game by MCTS" for "Planning algorithms in AI" class at Skoltech, Spring 2021.

The code is organized as follows: in the "main" branch there is a code for MCTS and MCTS-UCT algorithms. It is pretty easy and streighforward to run them via CLI.

For example, to train the tree, one could use:
`python run.py --save_name tree_lucky.pkl --use_UCT False`


To reproduce our experiments, you should simply run:
`sh experiment.sh`


There is another branch "Tree_rebuild", which contains the algorithm we introduced.
The way to launch those experiments is analogous.

-------

Here we present some MCTS policy examples, learned by algorithms we implemented:

1.   MCTS + Tree rebuild;
2.   MCTS;
3.   MCTS-UCT;


<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="480px" src="./gif/goal_reached_1.gif"></td>
    <td><img width="480px" src="./gif/goal_not_reached.gif"></td>
  </tr>

  <tr>
    <td><img width="480px" src="./gif/tree__use_UCT_False__seed_203.gif"></td>
    <td><img width="480px" src="./gif/tree__use_UCT_False__seed_200.gif"></td>
  </tr>
  <tr>
    <td><img width="480px" src="./gif/tree__use_UCT_True__seed_200.gif"></td>
    <td><img width="480px" src="./gif/tree__use_UCT_True__seed_201.gif"></td>
  </tr>


</table>
