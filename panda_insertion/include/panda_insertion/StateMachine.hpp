#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H
#include "panda_insertion/Controller.hpp"

enum State {
    Start,
    MoveToInitialPosition,
    InitialPosition,
    ExternalDownMovement,
    SpiralMotion,
    InternalDownMovement,
    Straightening,
    InsertionWiggle
};

class StateMachine
{
private:
    State activeState;
    Controller controller;

public:
    StateMachine();
    StateMachine(double loop_rate);
    void run();

    // States
    void start();
    void moveToInitialPosition();
    void initialPosition();
};

#endif