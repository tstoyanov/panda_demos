#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H
#include "panda_insertion/Controller.hpp"

enum State {
    Start,
    InitialPosition,
    ExternalDownMovement,
    SpiralMotion,
    InternalDownMovement,
    Straightening,
    InsertionWiggle,
    MoveToInitialPosition
};

class StateMachine
{
private:
    State activeState;
    Controller controller;

public:
    StateMachine();
    void run();

    // States
    void start();
    void initialPosition();
};

#endif