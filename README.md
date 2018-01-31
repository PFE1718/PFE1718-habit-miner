## Habit-miner skill
This skill is made to work with the full Habits Automation project https://github.com/PFE1718/mycroft-skills-automation.

Its role is to go through user logs and analyse them. By implementing machine learning algorithms, it can then detect most frequent user habits. They are then passed to the [habits-automation](https://github.com/PFE1718/mycroft-automation-handler) skill.

It can detect two types of habits :

* Time based habits (i.e launching the same skill regularly at the same time of the day)
* Group based habits (i.e launching a group of skills very frequently)

## Current state

## Working features:
 - time habits and group habits working and passed on to the habits automation system

Known issues:
 - 
 
## TODO:
 - Check that we are not adding two times the same habit to the system
 - Debug
