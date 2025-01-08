import torch

import dflex as df


class EvalRigidFKFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, model, joint_q):
        ctx.tape = df.adjoint.Tape()
        ctx.input = joint_q

        state_out = model.state()
        ctx.tape.launch(
            func=df.eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_joint_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                model.joint_X_pj,
                model.joint_X_cm,
                model.joint_axis
            ],
            outputs=[
                state_out.body_X_sc,
                state_out.body_X_sm
            ],
            adapter=model.adapter,
            preserve_output=True)

        ctx.outputs = df.adjoint.to_weak_list([state_out.body_X_sc, state_out.body_X_sm])

        return state_out.body_X_sc, state_out.body_X_sm

    @staticmethod
    def backward(ctx, *grads):

        # ensure grads are contiguous in memory
        adj_outputs = df.adjoint.make_contiguous(grads)

        # register outputs with tape
        outputs = df.adjoint.to_strong_list(ctx.outputs)
        for o in range(len(outputs)):
            ctx.tape.adjoints[outputs[o]] = adj_outputs[o]

        # replay launches backwards
        ctx.tape.replay()

        # find adjoint of inputs
        adj_inputs = []
        if ctx.input in ctx.tape.adjoints:
            adj_inputs.append(ctx.tape.adjoints[ctx.input])
        else:
            adj_inputs.append(None)

        # free the tape
        ctx.tape.reset()

        # filter grads to replace empty tensors / no grad / constant params with None
        return None, *df.adjoint.filter_grads(adj_inputs)
    
class EvalRigidIDFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, model, joint_q, joint_qd, body_X_sc, body_X_sm):
        ctx.tape = df.adjoint.Tape()
        ctx.input = df.adjoint.to_weak_list([joint_q, joint_qd, body_X_sc, body_X_sm])

        state_out = model.state()
        ctx.tape.launch(
            func=df.eval_rigid_id,
            dim=model.articulation_count,                       
            inputs=[
                model.articulation_joint_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                joint_qd,
                model.joint_axis,
                model.joint_target_ke,
                model.joint_target_kd,
                model.body_I_m,
                body_X_sc,
                body_X_sm,
                model.joint_X_pj,
                model.gravity
            ],
            outputs=[
                state_out.joint_S_s,
                state_out.body_I_s,
                state_out.body_v_s,
                state_out.body_f_s,
                state_out.body_a_s,
            ],
            adapter=model.adapter,
            preserve_output=True)

        ctx.outputs = df.adjoint.to_weak_list([state_out.joint_S_s, state_out.body_I_s, state_out.body_v_s, state_out.body_f_s, state_out.body_a_s])

        return state_out.joint_S_s, state_out.body_I_s, state_out.body_v_s, state_out.body_f_s, state_out.body_a_s
    
    @staticmethod
    def backward(ctx, *grads):

        # ensure grads are contiguous in memory
        adj_outputs = df.adjoint.make_contiguous(grads)

        # register outputs with tape
        outputs = df.adjoint.to_strong_list(ctx.outputs)
        for o in range(len(outputs)):
            ctx.tape.adjoints[outputs[o]] = adj_outputs[o]

        # replay launches backwards
        ctx.tape.replay()

        # find adjoint of inputs
        adj_inputs = []
        if ctx.input in ctx.tape.adjoints:
            adj_inputs.append(ctx.tape.adjoints[ctx.input])
        else:
            adj_inputs.append(None)

        # free the tape
        ctx.tape.reset()

        # filter grads to replace empty tensors / no grad / constant params with None
        return None, *df.adjoint.filter_grads(adj_inputs)

def eval_rigid_fk_grad(model: df.sim.Model, joint_q: torch.Tensor):
    """
    Forward kinematics to calculate body pos from joint info
    """
    return EvalRigidFKFunc.apply(model, joint_q)

def eval_rigid_id_grad(model: df.sim.Model, joint_q: torch.Tensor, joint_qd: torch.Tensor, body_X_sc: torch.Tensor, body_X_sm: torch.Tensor):
    """
    Forward kinematics to calculate body pos from joint info
    """
    return EvalRigidIDFunc.apply(model, joint_q, joint_qd, body_X_sc, body_X_sm)