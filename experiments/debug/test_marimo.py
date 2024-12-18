import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    from lexical_benchmark.datasets import analysis
    from lexical_benchmark.datasets import utils as dataset_utils

    return analysis, dataset_utils, mo


@app.cell
def __(analysis):
    analysis_data = analysis.LexicalBenchmarkDataset()
    return (analysis_data,)


@app.cell
def __(analysis_data, dataset_utils):
    childes_wf = analysis_data.childes_word_frequencies()

    chldes_wpf = dataset_utils.extend_wf_pos(childes_wf, "en_core_web_trf")
    analysis_data.childes_wfp_file.mk_parent()
    chldes_wpf.to_csv(analysis_data.childes_wfp_file, index=False)
    return childes_wf, chldes_wpf


@app.cell
def __(analysis_data, dataset_utils, stel_wf):
    stela_wf = analysis_data.stela_word_frequencies()
    stela_wpf = dataset_utils.extend_wf_pos(stel_wf, "en_core_web_trf")
    stela_wpf.to_csv(analysis_data.childes_wfp_file, index=False)
    return stela_wf, stela_wpf


@app.cell
def __(mo):
    mo.md(r"""# Information about dataset""")


if __name__ == "__main__":
    app.run()
