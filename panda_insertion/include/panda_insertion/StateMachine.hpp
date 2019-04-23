#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

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

public:
    StateMachine();
    void run();

    // States
    void start();
    void initialPosition();
};

#endif