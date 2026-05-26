use piria::{error::Result, PiriaRuntime};

fn main() -> Result<()> {
    let runtime = PiriaRuntime::open_persistent("node-local", ".piria")?;
    println!(
        "PIRIA single-node runtime initialized: node={}, orbs={}, events={}",
        runtime.node_id(),
        runtime.orb_count(),
        runtime.event_count()
    );
    Ok(())
}
