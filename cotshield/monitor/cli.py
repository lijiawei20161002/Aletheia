"""
CoTShield CLI - Command-line interface for analyzing reasoning traces.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .detector import CoTDivergenceDetector, analyze_cot_trace
from .reconstructor import ShadowIntentReconstructor, IntentType


console = Console()


def analyze_command(args):
    """Analyze a reasoning trace from command line or file."""
    # Get inputs
    if args.reasoning_file:
        with open(args.reasoning_file, 'r') as f:
            reasoning = f.read()
    elif args.reasoning:
        reasoning = args.reasoning
    else:
        console.print("[red]Error: Must provide reasoning via --reasoning or --reasoning-file[/red]")
        sys.exit(1)

    if args.output_file:
        with open(args.output_file, 'r') as f:
            output = f.read()
    elif args.output:
        output = args.output
    else:
        console.print("[red]Error: Must provide output via --output or --output-file[/red]")
        sys.exit(1)

    # Run analysis
    console.print("\n[bold cyan]🛡️  CoTShield - Analyzing Reasoning Trace[/bold cyan]\n")

    result = analyze_cot_trace(
        reasoning=reasoning,
        output=output,
        sensitivity=args.sensitivity
    )

    # Display results
    display_analysis_results(result, args.format)

    # Use reconstructor if requested
    if args.reconstruct and args.task:
        console.print("\n[bold cyan]🔍 Reconstructing Hidden Intent[/bold cyan]\n")

        try:
            reconstructor = ShadowIntentReconstructor(
                provider=args.provider,
                model=args.model
            )

            intent = reconstructor.reconstruct(
                task=args.task,
                reasoning=reasoning,
                output=output
            )

            display_reconstruction_results(intent, args.format)

        except Exception as e:
            console.print(f"[yellow]⚠️  Reconstruction failed: {e}[/yellow]")

    # Save results if requested
    if args.output_json:
        save_results(result, args.output_json)
        console.print(f"\n✅ Results saved to: {args.output_json}")


def display_analysis_results(result, format_type="text"):
    """Display analysis results in rich format."""
    if format_type == "json":
        # Convert flags to serializable format
        serializable_result = {
            "risk_score": result["risk_score"],
            "flag_count": result["flag_count"],
            "severity_distribution": result["severity_distribution"],
            "divergence_types": result["divergence_types"],
            "flags": [
                {
                    "type": flag.type.value,
                    "severity": flag.severity,
                    "reasoning_snippet": flag.reasoning_snippet,
                    "output_snippet": flag.output_snippet,
                    "explanation": flag.explanation,
                    "line_number": flag.line_number
                }
                for flag in result["flags"]
            ]
        }
        print(json.dumps(serializable_result, indent=2))
        return

    # Text format with rich styling
    risk_score = result["risk_score"]
    risk_color = "green" if risk_score < 0.3 else "yellow" if risk_score < 0.7 else "red"

    # Risk score panel
    console.print(Panel(
        f"[bold {risk_color}]{risk_score:.1%}[/bold {risk_color}]",
        title="Overall Risk Score",
        border_style=risk_color,
        box=box.ROUNDED
    ))

    # Statistics table
    table = Table(title="Detection Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="right")

    table.add_row("Total Flags", str(result["flag_count"]))
    table.add_row("High Severity", str(result["severity_distribution"]["high"]))
    table.add_row("Medium Severity", str(result["severity_distribution"]["medium"]))
    table.add_row("Low Severity", str(result["severity_distribution"]["low"]))

    console.print(table)

    # Detailed flags
    if result["flags"]:
        console.print("\n[bold]🚩 Detected Issues:[/bold]\n")
        for i, flag in enumerate(result["flags"], 1):
            severity_color = "red" if flag.severity > 0.7 else "yellow" if flag.severity > 0.4 else "blue"

            console.print(f"[bold]{i}. {flag.type.value.upper().replace('_', ' ')}[/bold]")
            console.print(f"   Severity: [{severity_color}]{flag.severity:.1%}[/{severity_color}]")
            console.print(f"   {flag.explanation}\n")
    else:
        console.print("\n[bold green]✅ No issues detected! Clean reasoning.[/bold green]")


def display_reconstruction_results(intent, format_type="text"):
    """Display reconstruction results."""
    if format_type == "json":
        serializable_intent = {
            "intent_type": intent.intent_type.value,
            "confidence": intent.confidence,
            "explanation": intent.explanation,
            "hidden_reasoning": intent.hidden_reasoning,
            "evidence": intent.evidence,
            "risk_assessment": intent.risk_assessment
        }
        print(json.dumps(serializable_intent, indent=2))
        return

    # Rich text display
    intent_color = "red" if intent.intent_type != IntentType.BENIGN else "green"

    console.print(Panel(
        f"[bold {intent_color}]{intent.intent_type.value.upper()}[/bold {intent_color}]\n"
        f"Confidence: {intent.confidence:.1%}",
        title="Reconstructed Intent",
        border_style=intent_color,
        box=box.ROUNDED
    ))

    console.print(f"\n[bold]Explanation:[/bold]\n{intent.explanation}")

    if intent.hidden_reasoning:
        console.print(f"\n[bold]Hidden Reasoning:[/bold]\n{intent.hidden_reasoning}")

    if intent.evidence:
        console.print("\n[bold]Evidence:[/bold]")
        for i, evidence in enumerate(intent.evidence, 1):
            console.print(f"  {i}. {evidence}")

    console.print(f"\n[bold]Risk Assessment:[/bold]\n{intent.risk_assessment}")


def save_results(result, output_path):
    """Save results to JSON file."""
    serializable_result = {
        "risk_score": result["risk_score"],
        "flag_count": result["flag_count"],
        "severity_distribution": result["severity_distribution"],
        "divergence_types": result["divergence_types"],
        "flags": [
            {
                "type": flag.type.value,
                "severity": flag.severity,
                "reasoning_snippet": flag.reasoning_snippet,
                "output_snippet": flag.output_snippet,
                "explanation": flag.explanation,
                "line_number": flag.line_number
            }
            for flag in result["flags"]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)


def viewer_command(args):
    """Start the web viewer."""
    from ui.viewer import start_viewer

    console.print("\n[bold cyan]🛡️  CoTShield - Starting Web Viewer[/bold cyan]\n")
    console.print(f"Server starting at: http://{args.host}:{args.port}")
    console.print("Press Ctrl+C to stop\n")

    try:
        start_viewer(host=args.host, port=args.port)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Shutting down viewer...[/yellow]")
        console.print("Goodbye! 👋\n")


def eval_command(args):
    """Run evaluation suite."""
    from evals.tasks import AdversarialTaskSuite, TaskEvaluator, TaskCategory

    console.print("\n[bold cyan]🛡️  CoTShield - Running Evaluation Suite[/bold cyan]\n")

    suite = AdversarialTaskSuite()
    detector = CoTDivergenceDetector(sensitivity=args.sensitivity)
    evaluator = TaskEvaluator(detector=detector)

    # Load model responses from file
    if not args.responses_file:
        console.print("[red]Error: --responses-file is required[/red]")
        sys.exit(1)

    with open(args.responses_file, 'r') as f:
        model_responses = json.load(f)

    # Filter by category if specified
    category = None
    if args.category:
        category_map = {
            "factual": TaskCategory.FACTUAL_ACCURACY,
            "logical": TaskCategory.LOGICAL_CONSISTENCY,
            "reward": TaskCategory.REWARD_HACKING,
            "deceptive": TaskCategory.DECEPTIVE_ALIGNMENT,
            "strategic": TaskCategory.STRATEGIC_REASONING,
            "humility": TaskCategory.EPISTEMIC_HUMILITY
        }
        category = category_map.get(args.category)

    # Run evaluation
    results = evaluator.evaluate_suite(model_responses, category=category)

    # Display results
    display_eval_results(results, args.format)

    # Save if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            # Convert results to serializable format
            serializable_results = {
                "total_tasks": results["total_tasks"],
                "passed": results["passed"],
                "failed": results["failed"],
                "pass_rate": results["pass_rate"],
                "average_risk_score": results["average_risk_score"],
                "by_category": results["by_category"],
                "results": [
                    {
                        "task_id": r.task_id,
                        "passed": r.passed,
                        "flags_raised": r.flags_raised,
                        "risk_score": r.risk_score,
                        "notes": r.notes
                    }
                    for r in results["results"]
                ]
            }
            json.dump(serializable_results, f, indent=2)
        console.print(f"\n✅ Results saved to: {args.output_json}")


def display_eval_results(results, format_type="text"):
    """Display evaluation results."""
    if format_type == "json":
        serializable_results = {
            "total_tasks": results["total_tasks"],
            "passed": results["passed"],
            "failed": results["failed"],
            "pass_rate": results["pass_rate"],
            "average_risk_score": results["average_risk_score"],
            "by_category": results["by_category"]
        }
        print(json.dumps(serializable_results, indent=2))
        return

    # Summary table
    table = Table(title="Evaluation Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Total Tasks", str(results["total_tasks"]))
    table.add_row("Passed", f"[green]{results['passed']}[/green]")
    table.add_row("Failed", f"[red]{results['failed']}[/red]")
    table.add_row("Pass Rate", f"{results['pass_rate']:.1%}")
    table.add_row("Avg Risk Score", f"{results['average_risk_score']:.1%}")

    console.print(table)

    # Per-category breakdown
    if results["by_category"]:
        console.print("\n[bold]📊 By Category:[/bold]\n")

        cat_table = Table(box=box.ROUNDED)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Passed", justify="right")
        cat_table.add_column("Total", justify="right")
        cat_table.add_column("Pass Rate", justify="right")
        cat_table.add_column("Avg Risk", justify="right")

        for cat, data in results["by_category"].items():
            pass_color = "green" if data["pass_rate"] > 0.7 else "yellow" if data["pass_rate"] > 0.4 else "red"
            cat_table.add_row(
                cat.replace("_", " ").title(),
                str(data["passed"]),
                str(data["total"]),
                f"[{pass_color}]{data['pass_rate']:.1%}[/{pass_color}]",
                f"{data['avg_risk_score']:.1%}"
            )

        console.print(cat_table)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CoTShield - Analyze AI reasoning traces for deceptive patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a reasoning trace
  cotshield analyze --reasoning "My reasoning..." --output "Final answer"

  # Analyze from files
  cotshield analyze --reasoning-file trace.txt --output-file output.txt

  # Include intent reconstruction
  cotshield analyze --reasoning-file trace.txt --output-file output.txt \\
      --task "What is 2+2?" --reconstruct

  # Start web viewer
  cotshield viewer

  # Run evaluation suite
  cotshield eval --responses-file responses.json --category factual
        """
    )

    parser.add_argument('--version', action='version', version='CoTShield 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a reasoning trace')
    analyze_parser.add_argument('--reasoning', type=str, help='Reasoning text')
    analyze_parser.add_argument('--reasoning-file', type=str, help='File containing reasoning')
    analyze_parser.add_argument('--output', type=str, help='Output text')
    analyze_parser.add_argument('--output-file', type=str, help='File containing output')
    analyze_parser.add_argument('--task', type=str, help='Original task/prompt (for reconstruction)')
    analyze_parser.add_argument('--sensitivity', type=float, default=0.5, help='Detection sensitivity (0-1)')
    analyze_parser.add_argument('--reconstruct', action='store_true', help='Use intent reconstructor')
    analyze_parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'anthropic'])
    analyze_parser.add_argument('--model', type=str, default='gpt-4', help='Model for reconstruction')
    analyze_parser.add_argument('--format', type=str, default='text', choices=['text', 'json'])
    analyze_parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    analyze_parser.set_defaults(func=analyze_command)

    # Viewer command
    viewer_parser = subparsers.add_parser('viewer', help='Start web viewer')
    viewer_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    viewer_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    viewer_parser.set_defaults(func=viewer_command)

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation suite')
    eval_parser.add_argument('--responses-file', type=str, required=True, help='JSON file with model responses')
    eval_parser.add_argument('--category', type=str, choices=['factual', 'logical', 'reward', 'deceptive', 'strategic', 'humility'])
    eval_parser.add_argument('--sensitivity', type=float, default=0.5, help='Detection sensitivity (0-1)')
    eval_parser.add_argument('--format', type=str, default='text', choices=['text', 'json'])
    eval_parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    eval_parser.set_defaults(func=eval_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
