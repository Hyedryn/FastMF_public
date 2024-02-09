import os
import sys

fastmf_path = os.path.join('D:\\', 'FastMF_python-paper')
if(not(fastmf_path in sys.path)):
    sys.path.insert(0,fastmf_path)

#%%
import fastmf.generation as gen
#%%
# Generate training and test data
scheme_file = "..\\data\\schemes\\scheme-HCPMGH_scheme.txt"
bvals_file = "..\\data\\schemes\\scheme-HCPMGH_bvals.txt"
dic_file = "..\\data\\dictionaries\\dictionary-fixedraddist_scheme-HCPMGH.mat"
synth_HCP_FixRadDist = gen.Synthetizer(scheme_file, bvals_file, dic_file, task_name="testFastMF",
                                       include_csf = True)

base_path = "..\\tests\\firstDataTest\\"
#%%
genFirstStep = True
if genFirstStep:
    # Generate training data
    synthStandard = synth_HCP_FixRadDist.generateStandardSet(1000, run_id="01")
    synthStandard.save(base_path, force_overwrite=True)
    genStandard = gen.Generator(synthStandard, base_path, 
                                orientation_estimate = 'CSD')
    genStandard.computeSphericalHarmonics()
    genStandard.computeNNLSWeights()
    genStandard.computeExhaustiveMF()

    synthStructured = synth_HCP_FixRadDist.generateStructuredSet(repetition=5, run_id="01")
    synthStructured.save(base_path)
    genStructured = gen.Generator(synthStructured, base_path, orientation_estimate = 'MSMTCSD')
    genStructured.computeSphericalHarmonics()
    genStructured.computeNNLSWeights()
    genStructured.computeExhaustiveMF()

#%%
formatterStandard = gen.DataFormatter(base_path, "testFastMF", "twoRuns", dic_file, 
                                      ["01", "02"], "standard", [1500, 250, 250])
formatterStructured = gen.DataFormatter(base_path, "testFastMF", "twoRuns", dic_file, 
                                        ["01", "02"], "structured", [800, 200, 200])

formatterStandard.genNNLSInput(normalization="None", orientation_estimate="CSD")
formatterStandard.genNNLSTarget(min_max_scaling=True)

formatterStandard.genSphericalHarmonicInput()
formatterStandard.genSphericalHarmonicTarget(min_max_scaling=False)


formatterStructured.genNNLSInput(normalization="None", orientation_estimate="MSMTCSD")
formatterStructured.genNNLSTarget(min_max_scaling=True)

formatterStructured.genSphericalHarmonicInput()
formatterStructured.genSphericalHarmonicTarget(min_max_scaling=False)