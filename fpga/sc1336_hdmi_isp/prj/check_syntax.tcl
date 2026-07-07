open_project ./sc1336_hdmi.xpr
update_compile_order -fileset sources_1
set top_name [get_property top [current_fileset]]
puts "INFO: top=$top_name"
puts "INFO: source_count=[llength [get_files -of_objects [get_filesets sources_1]]]"

if {[catch {check_syntax -fileset sources_1} result]} {
    puts "ERROR: check_syntax failed: $result"
    close_project
    exit 1
}

puts "INFO: check_syntax completed"
close_project
exit 0
