open_project ./sc1336_hdmi.xpr
reset_run synth_1
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1

set run_status [get_property STATUS [get_runs impl_1]]
puts "INFO: impl_1 status=$run_status"

open_run impl_1
report_timing_summary -max_paths 10 -report_unconstrained \
    -file ./sc1336_hdmi.runs/impl_1/sc1336_hdmi_timing_summary_routed_after_fix.rpt \
    -warn_on_violation
close_project
exit 0
