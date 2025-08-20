"""
This folder contains all the code to run the simulations. 

It starts with the definition of the objective functions and/or the functions 
to generate the extended SBML models. Afterwards we run and store all 
simulations.

Definition of the objective functions:

- problem_wrong_tp: pypesto Problem, as given in the original paper

- problem_marginalized_multi_tau: 
    For this objective, we assume that every data point has a different temporal 
    uncertainty. 
    We further fit SIGMA_TAU to the data and implement the MARGINALIZED MODEL.
- problem_marginalized_shared_tau:
    For this objective, we assume that all data points measured at the same time 
    point are shifted by the same value for tau i.e. temporal uncertainty. 
    We further fit SIGMA_TAU to the data and implement the MARGINALIZED MODEL.

- problem_full_multi_tau:
    For this objective, we assume that every data point has a different temporal 
    uncertainty. 
    We further fit SIGMA_TAU to the data and implement the FULLY EXTENDED MODEL.
- problem_full_shared_tau:
    For this objective, we assume that all data points measured at the same time 
    point are shifted by the same value for tau i.e. temporal uncertainty. 
    We further fit SIGMA_TAU to the data and implement the FULLY EXTENDED MODEL.
"""


import pypesto
import libsbml as sbml

import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter

"""
Create the constants & Setups, that are use throughout the simulations
"""
RANDOM_SEED = 0
optimimizer = optimize.FidesOptimizer()
sampler = None

# Create problem_wrong_tp
petab_yaml = "./Application_Bruno/petab_bruno_original/Bruno_JExpBot2016.yaml"

importer_wrong_tp = PetabImporter.from_yaml(petab_yaml)
problem_wrong_tp = importer_wrong_tp.create_problem(verbose=False)

result_wrong_tp = optimize.minimize(
    problem_wrong_tp, 
    optimizer=optimimizer,
    n_starts=N_STARTS
    )
sample.sample(problem_wrong_tp, n_samples=N_SAMPLES, result=result_wrong_tp)

# Create problem_marginalized_multi_tau
def _create_parameter(
    sbml_model, 
    parameter_id: str, 
    parameter_name: str, 
    parameter_value: float,
):
    """
    Create a new parameter in the SBML model. Returns the modified model + parameter object.
    
    Parameters:
    ----------
    sbml_model : libsbml.Model
        The SBML model to modify.
    parameter_id : str
        The ID of the parameter to create.
    parameter_name : str
        The name of the parameter to create.
    parameter_value : float
        The value of the parameter to create.
    
    Returns
    -------
    sbml_model : libsbml.Model
        The modified SBML model with the new parameter.
    k : libsbml.Parameter
        The created parameter object.
    """
    k = model.createParameter()
    k.setId(parameter_id)
    k.setName(parameter_name)
    k.setConstant(True)
    k.setValue(parameter_value)
    
    return sbml_model, k

def create_marginalized_sbml_multi_tau(
    sbml_input_dir: str,
    sbml_input_dir: str,
):
    """
    Create the extended SBML model with the given parameters.

    Parameters
    ----------
    sbml_input_dir : str
        Path to the input SBML file.
    sbml_input_dir : str
        Directory to save the generated SBML file.
    """
    # Load the SBML model
    sbml_model = sbml.readSBML(sbml_input_dir).getModel()

    # create a time point parameter, that can be used furing integration 

    #Create a new parameter for z
    sbml_model = _create_parameter(
        sbml_model, 
        parameter_id="z", 
        parameter_name="z", 
        parameter_value=0.0
    )
    
    # create the function for the right hand side of the ODEs

    # create a rate rule, that defines the right hand side for z

    rate_rule = sbml_model.

    def _generate_z(t_i, d_i, state_id):
        """
        Generate the z value for the given time point and data point.

        Parameters
        ----------
        t_i : float
            The time point.
        d_i : float
            The data point.
        state_id : str
            The state ID.

        Returns
        -------
        float
            The generated z value.
        """
        pass

    

    # Save the extended SBML model to a file
    sbml.writeSBMLToFile(sbml_model, sbml_output_dir)

    return extended_model