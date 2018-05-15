
#include "acados_cpp/ocp_nlp/ocp_nlp.hpp"

#include <algorithm>
#include <functional>
#include <string>

#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"
#include "acados/ocp_nlp/ocp_nlp_cost_nls.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_cont.h"
#include "acados/ocp_nlp/ocp_nlp_sqp.h"

#include "acados_c/external_function_interface.h"
#include "acados_cpp/ocp_bounds.hpp"
#include "acados_cpp/ocp_dimensions.hpp"
#include "acados_cpp/utils.hpp"

#include "blasfeo/include/blasfeo_d_aux.h"

#include "casadi/casadi.hpp"
#include "casadi/mem.h"

#if (defined _WIN32 || defined _WIN64 || defined __MINGW32__ || defined __MINGW64__)
#include <windows.h>
const char dynamic_library_suffix[] = ".dll";
static void create_directory(std::string name){
    CreateDirectory(name.c_str(), NULL);
    if (ERROR_ALREADY_EXISTS != GetLastError())
        throw std::runtime_error("Could not create folder '" + name + "'.");
}
#else
#include <dlfcn.h>
const char dynamic_library_suffix[] = ".so";
static void create_directory(std::string name){
    std::string command = "mkdir -p " + name;
    int creation_failed = system(command.c_str());
    if (creation_failed)
        throw std::runtime_error("Could not create folder '" + name + "'.");
}
#endif
const char tmp_dir[] = "_casadi_gen";

namespace acados
{
using std::map;
using std::vector;
using std::string;

ocp_nlp::ocp_nlp(std::vector<int> nx, std::vector<int> nu, std::vector<int> ng, std::vector<int> nh,
                 std::vector<int> ns)
    : N(nx.size() - 1), needs_initializing_(true)
{
    // Number of controls on last stage should be zero;
    if (!nu.empty()) nu.back() = 0;

    std::map<std::string, std::vector<int>> dims{{"nx", nx}, {"nu", nu}, {"nbx", nx}, {"nbu", nu},
                                                 {"ng", ng}, {"nh", nh}, {"ns", ns}};

    if (!are_valid_ocp_dimensions(dims, {"nx", "nu", "nbx", "nbu", "nb", "ng", "nh", "ns"}))
        throw std::invalid_argument("Invalid dimensions map.");

    d_["nx"] = nx;
    d_["nu"] = nu;
    d_["nbx"] = nx;
    d_["nbu"] = nu;
    d_["ng"] = ng;
    d_["nh"] = nh;
    d_["ns"] = ns;
    d_["ny"] = vector<int>(N+1);

    int config_size = ocp_nlp_solver_config_calculate_size(N);
    void *raw_memory = malloc(config_size);
    config_.reset(ocp_nlp_solver_config_assign(N, raw_memory));

    for (int i = 0; i <= N; ++i)
        ocp_nlp_constraints_config_initialize_default(config_->constraints[i]);

    int nlp_size = ocp_nlp_in_calculate_size_self(N);
    raw_memory = malloc(nlp_size);
    nlp_.reset(ocp_nlp_in_assign_self(N, raw_memory));

    int dims_size = ocp_nlp_dims_calculate_size_self(N);
    raw_memory = malloc(dims_size);
    dims_.reset(ocp_nlp_dims_assign_self(N, raw_memory));

    for (int i = 0; i <= N; ++i)
    {
        dims_->nv[i] = d_["nx"][i] + d_["nu"][i] + d_["ns"][i];
        dims_->nx[i] = d_["nx"][i];
        dims_->nu[i] = d_["nu"][i];
        dims_->ni[i] = d_["nbx"][i] + d_["nbu"][i] + d_["ng"][i] +
                       d_["nh"][i];
    }

    for (int i = 0; i <= N; ++i)
    {
        dims_->constraints[i] = malloc(sizeof(ocp_nlp_constraints_dims));
        ocp_nlp_constraints_dims *con_dims = (ocp_nlp_constraints_dims *) dims_->constraints[i];
        con_dims->nx = d_["nx"][i];
        con_dims->nu = d_["nu"][i];
        con_dims->nb = d_["nbx"][i] + d_["nbu"][i];
        con_dims->nbx = d_["nbx"][i];
        con_dims->nbu = d_["nbu"][i];
        con_dims->ng = d_["ng"][i];
        con_dims->nh = d_["nh"][i];
        con_dims->np = 0;
        con_dims->ns = d_["ns"][i];
        int num_bytes = ocp_nlp_constraints_model_calculate_size(config_->constraints[i], con_dims);
        void *raw_memory = calloc(1, num_bytes);
        nlp_->constraints[i] =
            ocp_nlp_constraints_model_assign(config_->constraints[i], con_dims, raw_memory);
    }

    int qp_dims_size = ocp_qp_dims_calculate_size(N);
    raw_memory = malloc(qp_dims_size);
    dims_->qp_solver = ocp_qp_dims_assign(N, raw_memory);

    for (int stage = 0; stage <= N; ++stage)
    {
        cached_bounds["lbx"].push_back(vector<double>(nx[stage], -INFINITY));
        cached_bounds["ubx"].push_back(vector<double>(nx[stage], +INFINITY));
        cached_bounds["lbu"].push_back(vector<double>(nu[stage], -INFINITY));
        cached_bounds["ubu"].push_back(vector<double>(nu[stage], +INFINITY));
    }
}

ocp_nlp::ocp_nlp(int N, int nx, int nu, int ng, int nh, int ns)
    : ocp_nlp(std::vector<int>(N + 1, nx), std::vector<int>(N + 1, nu), std::vector<int>(N + 1, ng),
              std::vector<int>(N + 1, nh), std::vector<int>(N + 1, ns))
{
}

void ocp_nlp::initialize_solver(std::string solver_name, std::map<std::string, option_t *> options)
{
    if (solver_name == "sqp")
        ocp_nlp_sqp_config_initialize_default(config_.get());
    else
        throw std::invalid_argument("Solver name '" + solver_name + "' not known.");

    string qp_solver_name;
    if (options.count("qp_solver"))
        qp_solver_name = to_string(options.at("qp_solver"));
    else
        qp_solver_name = "hpipm";

    if (qp_solver_name == "hpipm")
        ocp_qp_xcond_solver_config_initialize_default(PARTIAL_CONDENSING_HPIPM, config_->qp_solver);
    else if (qp_solver_name == "qpoases")
        ocp_qp_xcond_solver_config_initialize_default(FULL_CONDENSING_QPOASES, config_->qp_solver);
    else
        throw std::invalid_argument("QP solver name '" + qp_solver_name + "' not known.");

    squeeze_dimensions(cached_bounds);

    ocp_nlp_dims_initialize(config_.get(), d_["nx"].data(), d_["nu"].data(),
                            d_["ny"].data(), d_["nbx"].data(),
                            d_["nbu"].data(), d_["ng"].data(),
                            d_["nh"].data(), std::vector<int>(N + 1, 0).data(),
                            d_["ns"].data(), dims_.get());

    solver_options_.reset(ocp_nlp_opts_create(config_.get(), dims_.get()));
}

ocp_nlp_solution ocp_nlp::solve()
{
    fill_bounds(cached_bounds);

    solver_.reset(ocp_nlp_create(config_.get(), dims_.get(), solver_options_.get()));

    auto result = std::shared_ptr<ocp_nlp_out>(ocp_nlp_out_create(config_.get(), dims_.get()));

    ocp_nlp_solve(solver_.get(), nlp_.get(), result.get());

    return ocp_nlp_solution(result, dims_);
}

void ocp_nlp::set_field(string field, vector<double> v)
{
    for (int stage = 0; stage <= N; ++stage) set_field(field, stage, v);
}

void ocp_nlp::set_field(string field, int stage, vector<double> v)
{
    if (field == "lbx" || field == "ubx")
    {
        if (v.size() != (size_t) d_["nx"].at(stage))
            throw std::invalid_argument("Expected size " +
                                        std::to_string(d_["nx"].at(stage)) + " but got " +
                                        std::to_string(v.size()) + " instead.");

        cached_bounds[field].at(stage) = v;
    }
    else if (field == "lbu" || field == "ubu")
    {
        if (v.size() != (size_t) d_["nu"].at(stage))
            throw std::invalid_argument("Expected size " +
                                        std::to_string(d_["nu"].at(stage)) + " but got " +
                                        std::to_string(v.size()) + " instead.");

        cached_bounds[field].at(stage) = v;
    }
    else
    {
        throw std::invalid_argument("OCP NLP does not contain field '" + field + "'.");
    }
}

static bool is_valid_model(const casadi::Function &model)
{
    if (model.n_in() != 2)
        throw std::invalid_argument("An ODE model should have 2 inputs: states and controls.");
    if (model.n_out() != 1)
        throw std::runtime_error("An ODE model should have 1 output: the right hand side");

    casadi::SX x = model.sx_in(0);
    casadi::SX u = model.sx_in(1);
    int_t nx = x.size1();
    std::vector<casadi::SX> input{x, u};

    casadi::SX rhs = casadi::SX::vertcat(model(input));
    if (rhs.size1() != nx)
        throw std::runtime_error("Length of right hand size should equal number of states");
    return true;
}

static bool is_valid_nls_residual(const casadi::Function &r)
{
    if (r.n_in() != 1 && r.n_in() != 2)
        throw std::invalid_argument("An NLS residual should have max 2 inputs: states and "
                                    "controls.");
    if (r.n_out() != 1)
        throw std::invalid_argument("An NLS residual function should have 1 output.");

    return true;
}

void ocp_nlp::set_stage_cost(std::vector<double> C, std::vector<double> y_ref,
                             std::vector<double> W)
{
    for (int i = 0; i < N; ++i)
        set_stage_cost(i, C, y_ref, W);
}

void ocp_nlp::set_terminal_cost(std::vector<double> C, std::vector<double> y_ref,
                                std::vector<double> W)
{
    set_stage_cost(N, C, y_ref, W);
}

void ocp_nlp::set_stage_cost(int stage, std::vector<double> C, std::vector<double> y_ref,
                             std::vector<double> W)
{
    if (C.size() % (d_["nx"][stage]+d_["nu"][stage]+d_["ns"][stage]) != 0)
        throw std::invalid_argument("Linear least squares matrix has wrong dimensions.");

    auto ny = C.size() / (d_["nx"][stage]+d_["nu"][stage]);
    d_["ny"][stage] = ny;

    if (W.size() != ny*ny)
        throw std::invalid_argument("Linear least squares weighting matrix has wrong dimensions.");

    ocp_nlp_cost_ls_config_initialize_default(config_->cost[stage]);

    int nx = d_["nx"][stage], nu = d_["nu"][stage];

    dims_->cost[stage] = malloc(sizeof(ocp_nlp_cost_ls_dims));
    ocp_nlp_cost_ls_dims *cost_dims = (ocp_nlp_cost_ls_dims *) dims_->cost[stage];
    cost_dims->nx = nx;
    cost_dims->nu = nu;
    cost_dims->ns = d_["ns"][stage];
    cost_dims->ny = ny;
    int num_bytes = ocp_nlp_cost_ls_model_calculate_size(config_->cost[stage], cost_dims);
    void *raw_memory = calloc(1, num_bytes);
    nlp_->cost[stage] = ocp_nlp_cost_ls_model_assign(config_->cost[stage], cost_dims, raw_memory);

    ocp_nlp_cost_ls_model *stage_cost_ls = (ocp_nlp_cost_ls_model *) nlp_->cost[stage];

    // Cyt
    blasfeo_pack_tran_dmat(ny, nu, &C.data()[ny*nx], ny, &stage_cost_ls->Cyt, 0, 0);
    blasfeo_pack_tran_dmat(ny, nx, C.data(), ny, &stage_cost_ls->Cyt, nu, 0);

    // y_ref
    blasfeo_pack_dvec(ny, y_ref.data(), &stage_cost_ls->y_ref, 0);

    // W
    blasfeo_pack_dmat(ny, ny, W.data(), ny, &stage_cost_ls->W, 0, 0);

}

void ocp_nlp::set_stage_cost(const casadi::Function& r, vector<double> y_ref, vector<double> W)
{
    for (int i = 0; i < N; ++i)
        set_stage_cost(i, r, y_ref, W);
}

void ocp_nlp::set_stage_cost(int stage, const casadi::Function& r, vector<double> y_ref,
                             vector<double> W)
{
    if (!is_valid_nls_residual(r))
        throw std::invalid_argument("Invalid NLS residual function.");

    int ny = r.numel_out(0);

    if (W.size() != ny*ny)
        throw std::invalid_argument("Linear least squares weighting matrix has wrong dimensions.");

    d_["ny"][stage] = ny;

    ocp_nlp_cost_nls_config_initialize_default(config_->cost[stage]);

    int dims_size = ocp_nlp_cost_nls_dims_calculate_size(config_->cost[stage]);
    void *raw_memory = malloc(dims_size);
    dims_->cost[stage] = ocp_nlp_cost_nls_dims_assign(config_->cost[stage], raw_memory);
    ocp_nlp_cost_nls_dims_initialize(config_->cost[stage], dims_->cost[stage],
                                     d_["nx"][stage], d_["nu"][stage], d_["ny"][stage],
                                     d_["ns"][stage]);

    int model_size = ocp_nlp_cost_nls_model_calculate_size(config_->cost[stage],
                                                           dims_->cost[stage]);
    raw_memory = malloc(model_size);
    nlp_->cost[stage] = ocp_nlp_cost_nls_model_assign(config_->cost[stage], dims_->cost[stage],
                                                      raw_memory);

    auto residual_functions = create_nls_residual(r);

    generate_casadi_function(residual_functions);

    for (auto& f : residual_functions)
        residuals_handle_[f.first] = compile_and_load(f.first);

    nls_residual_.casadi_fun = reinterpret_cast<casadi_eval_t>(
        load_function("nls_res", residuals_handle_["nls_res"]));
    nls_residual_.casadi_n_in = reinterpret_cast<casadi_getint_t>(
        load_function("nls_res_n_in", residuals_handle_["nls_res"]));
    nls_residual_.casadi_n_out = reinterpret_cast<casadi_getint_t>(
        load_function("nls_res_n_out", residuals_handle_["nls_res"]));
    nls_residual_.casadi_sparsity_in = reinterpret_cast<casadi_sparsity_t>(
        load_function("nls_res_sparsity_in", residuals_handle_["nls_res"]));
    nls_residual_.casadi_sparsity_out = reinterpret_cast<casadi_sparsity_t>(
        load_function("nls_res_sparsity_out", residuals_handle_["nls_res"]));
    nls_residual_.casadi_work = reinterpret_cast<casadi_work_t>(
        load_function("nls_res_work", residuals_handle_["nls_res"]));

    external_function_casadi_create(&nls_residual_);

    ocp_nlp_cost_nls_model *model = (ocp_nlp_cost_nls_model *) nlp_->cost[stage];
    model->nls_jac = (external_function_generic *) &nls_residual_;
    blasfeo_pack_dmat(ny, ny, W.data(), ny, &model->W, 0, 0);
    blasfeo_pack_dvec(ny, y_ref.data(), &model->y_ref, 0);
}

void ocp_nlp::set_terminal_cost(const casadi::Function& r, vector<double> y_ref, vector<double> W)
{
    set_stage_cost(N, r, y_ref, W);
}

void ocp_nlp::set_dynamics(const casadi::Function &model, std::map<std::string, option_t *> options)
{
    if (!is_valid_model(model)) throw std::invalid_argument("Model is invalid.");

    if (!options.count("step")) throw std::invalid_argument("Expected 'step' as an option.");

    std::fill(nlp_->Ts, nlp_->Ts+N, to_double(options["step"]));

    sim_solver_plan sim_plan;
    if (to_string(options.at("integrator")) == "rk4")
        sim_plan.sim_solver = ERK;
    else
        throw std::invalid_argument("Invalid integrator.");

    for (int i = 0; i < N; ++i)
    {
        ocp_nlp_dynamics_cont_config_initialize_default(config_->dynamics[i]);
        config_->dynamics[i]->sim_solver = sim_config_create(sim_plan);

        int dims_size = ocp_nlp_dynamics_cont_dims_calculate_size(config_->dynamics[i]);
        void *raw_memory = malloc(dims_size);
        dims_->dynamics[i] = ocp_nlp_dynamics_cont_dims_assign(config_->dynamics[i], raw_memory);
        ocp_nlp_dynamics_cont_dims_initialize(config_->dynamics[i], dims_->dynamics[i],
                                              d_["nx"][i], d_["nu"][i],
                                              d_["nx"][i + 1], d_["nu"][i + 1]);

        int model_size =
            ocp_nlp_dynamics_cont_model_calculate_size(config_->dynamics[i], dims_->dynamics[i]);
        raw_memory = malloc(model_size);
        nlp_->dynamics[i] = ocp_nlp_dynamics_cont_model_assign(config_->dynamics[i],
                                                               dims_->dynamics[i], raw_memory);
    }

    auto functions = create_explicit_ode_functions(model);

    generate_casadi_function(functions);

    for (auto& elem : functions)
        dynamics_handle_[elem.first] = compile_and_load(elem.first);

    forw_vde_.casadi_fun = reinterpret_cast<casadi_eval_t>(
        load_function("expl_vde_for", dynamics_handle_["expl_vde_for"]));
    forw_vde_.casadi_n_in = reinterpret_cast<casadi_getint_t>(
        load_function("expl_vde_for_n_in", dynamics_handle_["expl_vde_for"]));
    forw_vde_.casadi_n_out = reinterpret_cast<casadi_getint_t>(
        load_function("expl_vde_for_n_out", dynamics_handle_["expl_vde_for"]));
    forw_vde_.casadi_sparsity_in = reinterpret_cast<casadi_sparsity_t>(
        load_function("expl_vde_for_sparsity_in", dynamics_handle_["expl_vde_for"]));
    forw_vde_.casadi_sparsity_out = reinterpret_cast<casadi_sparsity_t>(
        load_function("expl_vde_for_sparsity_out", dynamics_handle_["expl_vde_for"]));
    forw_vde_.casadi_work = reinterpret_cast<casadi_work_t>(
        load_function("expl_vde_for_work", dynamics_handle_["expl_vde_for"]));

    external_function_casadi_create(&forw_vde_);

    for (int stage = 0; stage < N; ++stage)
        nlp_set_model_in_stage(config_.get(), nlp_.get(), stage, "expl_vde_for", &forw_vde_);
};

void generate_casadi_function(map<string, casadi::Function> functions)
{
    create_directory(tmp_dir);

    for (auto& elem : functions)
    {
        auto generator = casadi::CodeGenerator(elem.first + ".c", {
                                                                  {"with_header", true},
                                                                  {"with_export", false},
                                                                  {"casadi_int", "int"}});
        generator.add(elem.second);
        generator.generate(string(tmp_dir) + "/");
    }
}

void *load_function(std::string function_name, void *handle)
{
#if (defined _WIN32 || defined _WIN64 || defined __MINGW32__ || defined __MINGW64__)
    return GetProcAddress((HMODULE) handle, function_name.c_str());
#else
    return dlsym(handle, function_name.c_str());
#endif
}

map<string, casadi::Function> create_nls_residual(const casadi::Function& r)
{
    casadi::SX x = r.sx_in(0);
    casadi::SX u = r.sx_in(1);

    vector<casadi::SX> xu {x, u};
    vector<casadi::SX> ux {u, x};

    casadi::SX r_new = casadi::SX::vertcat(r(xu));
    casadi::SX r_jacT = casadi::SX::jacobian(r_new, casadi::SX::vertcat(ux)).T();

    casadi::Function r_fun("nls_res", {casadi::SX::vertcat(ux)}, {r_new, r_jacT});

    return {{"nls_res", r_fun}};
}

map<string, casadi::Function> create_explicit_ode_functions(
    const casadi::Function &model)
{
    casadi::SX x = model.sx_in(0);
    casadi::SX u = model.sx_in(1);
    int_t nx = x.size1();
    int_t nu = u.size1();
    casadi::SX rhs = casadi::SX::vertcat(model(vector<casadi::SX>({x, u})));

    casadi::SX Sx = casadi::SX::sym("Sx", nx, nx);
    casadi::SX Su = casadi::SX::sym("Su", nx, nu);

    casadi::SX vde_x = casadi::SX::jtimes(rhs, x, Sx);
    casadi::SX vde_u = casadi::SX::jacobian(rhs, u) + casadi::SX::jtimes(rhs, x, Su);

    casadi::Function vde_fun("expl_vde_for", {x, Sx, Su, u}, {rhs, vde_x, vde_u});

    casadi::SX jac_x = casadi::SX::zeros(nx, nx) + casadi::SX::jacobian(rhs, x);
    casadi::Function jac_fun("expl_ode_jac" + model.name(), {x, Sx, Su, u}, {rhs, jac_x});

    return {{"expl_vde_for", vde_fun}, {"expl_ode_jac", jac_fun}};
}

void *compile_and_load(std::string name)
{
#if (defined _WIN32 || defined _WIN64 || defined __MINGW32__ || defined __MINGW64__)
    std::string compiler{"mex"};
#else
    std::string compiler{"cc"};
#endif
    void *handle;
    std::string library_name = name + dynamic_library_suffix;
    std::string path_to_library = "./" + string(tmp_dir) + "/" + library_name;
    std::string path_to_file = "./" + string(tmp_dir) + "/" + name;
    char command[MAX_STR_LEN];
    snprintf(command, sizeof(command), "%s -fPIC -shared -g %s.c -o %s", compiler.c_str(),
             path_to_file.c_str(), path_to_library.c_str());
    int compilation_failed = system(command);
    if (compilation_failed)
        throw std::runtime_error("Something went wrong when compiling the model.");
#if (defined _WIN32 || defined _WIN64 || defined __MINGW32__ || defined __MINGW64__)
    handle = LoadLibrary(path_to_library.c_str());
#else
    handle = dlopen(path_to_library.c_str(), RTLD_LAZY);
#endif
    if (handle == NULL)
        throw std::runtime_error("Loading of " + path_to_library +
                                 " failed. Error message: " + load_error_message());
    return handle;
}

void ocp_nlp::set_bound(std::string bound, int stage, std::vector<double> new_bound)
{
    ocp_nlp_constraints_model **constraints = (ocp_nlp_constraints_model **) nlp_->constraints;
    ocp_nlp_constraints_dims **constraint_dims = (ocp_nlp_constraints_dims **) dims_->constraints;

    int nbx = constraint_dims[stage]->nbx;
    int nbu = constraint_dims[stage]->nbu;
    int ni = dims_->ni[stage];

    if (bound == "lbx")
    {
        blasfeo_pack_dvec(nbx, new_bound.data(), &constraints[stage]->d, nbu);
    }
    else if (bound == "ubx")
    {
        blasfeo_pack_dvec(nbx, new_bound.data(), &constraints[stage]->d, ni + nbu);
    }
    else if (bound == "lbu")
    {
        blasfeo_pack_dvec(nbu, new_bound.data(), &constraints[stage]->d, 0);
    }
    else if (bound == "ubu")
    {
        blasfeo_pack_dvec(nbu, new_bound.data(), &constraints[stage]->d, ni);
    }
    else
    {
        throw std::invalid_argument("Expected one of {'lbx', 'ubx', 'lbu', 'ubu'} but got " +
                                    bound + ".");
    }
}

void ocp_nlp::change_bound_dimensions(std::vector<int> nbx, std::vector<int> nbu)
{
    if (nbx.size() != (size_t)(num_stages() + 1) || nbu.size() != (size_t)(num_stages() + 1))
        throw std::invalid_argument("Expected " + std::to_string(num_stages()) + " bounds.");

    ocp_nlp_constraints_dims **constraint_dims = (ocp_nlp_constraints_dims **) dims_->constraints;

    for (int i = 0; i <= N; ++i)
    {
        constraint_dims[i]->nbx = nbx[i];
        constraint_dims[i]->nbu = nbu[i];
    }

    d_["nbx"] = nbx;
    d_["nbu"] = nbu;
}

bool ocp_nlp::needs_initializing() { return needs_initializing_; }

void ocp_nlp::needs_initializing(bool flag) { needs_initializing_ = flag; }

vector<int> ocp_nlp::get_bound_indices(std::string bound, int stage)
{
    ocp_nlp_constraints_model **constraints = (ocp_nlp_constraints_model **) nlp_->constraints;
    ocp_nlp_constraints_dims **constraint_dims = (ocp_nlp_constraints_dims **) dims_->constraints;

    vector<int> idxb;

    int nbx = constraint_dims[stage]->nbx;
    int nbu = constraint_dims[stage]->nbu;
    int nu = dims_->nu[stage];

    if (bound == "lbx" || bound == "ubx")
    {
        for (int i = 0; i < nbx; ++i) idxb.push_back(constraints[stage]->idxb[nbu + i] - nu);
    }
    else if (bound == "lbu" || bound == "ubu")
    {
        for (int i = 0; i < nbu; ++i) idxb.push_back(constraints[stage]->idxb[i]);
    }
    else
    {
        throw std::invalid_argument("Expected 'lbx', 'ubx', 'lbu' or 'ubu' but got '" + bound +
                                    "'.");
    }

    return idxb;
}

void ocp_nlp::set_bound_indices(std::string bound, int stage, std::vector<int> idxb)
{
    ocp_nlp_constraints_model **constraints = (ocp_nlp_constraints_model **) nlp_->constraints;
    ocp_nlp_constraints_dims **constraint_dims = (ocp_nlp_constraints_dims **) dims_->constraints;

    int nbx = constraint_dims[stage]->nbx;
    int nbu = constraint_dims[stage]->nbu;

    if (bound == "x")
    {
        if (idxb.size() != (size_t) nbx)
            throw std::invalid_argument("Expected " + std::to_string(constraint_dims[stage]->nbx) +
                                        " bound indices but got " + std::to_string(idxb.size()) +
                                        ".");
        for (int i = 0; i < nbx; ++i)
            constraints[stage]->idxb[nbu + i] = dims_->nu[stage] + idxb[i];
    }
    else if (bound == "u")
    {
        if (idxb.size() != (size_t) nbu)
            throw std::invalid_argument("Expected " + std::to_string(constraint_dims[stage]->nbu) +
                                        " bound indices but got " + std::to_string(idxb.size()) +
                                        ".");
        for (int i = 0; i < nbu; ++i) constraints[stage]->idxb[i] = idxb[i];
    }
    else
    {
        throw std::invalid_argument("Can only get bound indices for 'x' and 'u'.");
    }
}

int ocp_nlp::num_stages() { return N; }

ocp_nlp::~ocp_nlp()
{
    for (int i = 0; i < N; ++i)
    {
        free(dims_->constraints[i]);
        free(dims_->cost[i]);
        free(dims_->dynamics[i]);

        free(nlp_->constraints[i]);
        free(nlp_->cost[i]);
        free(nlp_->dynamics[i]);

        free(config_->dynamics[i]->sim_solver);
    }

    free(dims_->constraints[N]);
    free(dims_->cost[N]);

    free(nlp_->constraints[N]);
    free(nlp_->cost[N]);

    free(forw_vde_.ptr_ext_mem);

    for (auto &elem : dynamics_handle_) dlclose(elem.second);
}

}  // namespace acados