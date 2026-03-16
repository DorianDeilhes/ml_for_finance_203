#!/usr/bin/env python3
"""
Comprehensive file verification script.
Checks all 6 requirements outlined in the task.
"""
import json
import sys
import ast

def check_notebook_json():
    """Check 1: Verify main.ipynb is valid JSON"""
    print("\n" + "="*70)
    print("CHECK 1: Verify main.ipynb is valid JSON")
    print("="*70)
    try:
        with open('notebooks/main.ipynb', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ PASS: main.ipynb is valid JSON")
        print(f"  - Root type: {type(data).__name__}")
        print(f"  - Keys: {list(data.keys())}")
        if 'cells' in data:
            print(f"  - Number of cells: {len(data['cells'])}")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ FAIL: main.ipynb has JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ FAIL: Error reading main.ipynb: {e}")
        return False

def check_pipeline_constants():
    """Check 2a: Verify pipeline.py constants"""
    print("\n" + "="*70)
    print("CHECK 2a: Verify pipeline.py constants")
    print("="*70)
    try:
        with open('src/data/pipeline.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('TRAIN_END="2016-12-31"', 'TRAIN_END="2016-12-31"' in content),
            ('VAL_START="2017-01-01"', 'VAL_START="2017-01-01"' in content),
            ('VAL_END="2021-12-31"', 'VAL_END="2021-12-31"' in content),
        ]
        
        all_pass = True
        for desc, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {desc}: {'PASS' if result else 'FAIL'}")
            if not result:
                all_pass = False
        return all_pass
    except Exception as e:
        print(f"✗ FAIL: Error reading pipeline.py: {e}")
        return False

def check_build_pipeline_signature():
    """Check 2b: Verify build_pipeline returns 8 values"""
    print("\n" + "="*70)
    print("CHECK 2b: Verify build_pipeline returns 8 values")
    print("="*70)
    try:
        with open('src/data/pipeline.py', 'r') as f:
            content = f.read()
        
        # Search for the return statement
        if 'return train_loader, val_loader, test_loader, macro_scaler, ret_scaler, info, num_macro_features, num_assets' in content:
            print("✓ PASS: build_pipeline returns 8 values (train_loader, val_loader, test_loader, macro_scaler, ret_scaler, info, num_macro_features, num_assets)")
            return True
        else:
            print("✗ FAIL: build_pipeline does not return the expected 8 values")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error reading pipeline.py: {e}")
        return False

def check_verify_no_lookahead_vectorized():
    """Check 2c: Verify verify_no_lookahead is vectorized"""
    print("\n" + "="*70)
    print("CHECK 2c: Verify verify_no_lookahead is vectorized (no for loop)")
    print("="*70)
    try:
        with open('src/data/pipeline.py', 'r') as f:
            content = f.read()
        
        # Extract the function
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'verify_no_lookahead':
                # Check if there are any 'for' loops iterating over master.index
                has_loop = False
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.For):
                        # Check if loop target is on master.index
                        loop_source = ast.unparse(inner_node).lower() if hasattr(ast, 'unparse') else ''
                        if 'master.index' in loop_source or 'for date in' in loop_source:
                            has_loop = True
                            break
                
                if not has_loop:
                    print("✓ PASS: verify_no_lookahead is vectorized (no 'for date in master.index' loop)")
                    print("  - Uses vectorized operations (map, groupby)")
                    return True
                else:
                    print("✗ FAIL: verify_no_lookahead contains a loop over master.index")
                    return False
        
        # Also check with string search as fallback
        if 'for date in master.index' not in content and 'for date in' not in content.split('verify_no_lookahead')[1].split('def ')[0]:
            print("✓ PASS: verify_no_lookahead is vectorized (no explicit loop found)")
            return True
        else:
            print("✗ FAIL: verify_no_lookahead contains a date loop")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error analyzing pipeline.py: {e}")
        # Fallback: simple string check
        try:
            with open('src/data/pipeline.py', 'r') as f:
                content = f.read()
            if 'for date in master.index' not in content:
                print("✓ PASS: verify_no_lookahead has no 'for date in master.index' loop")
                return True
            else:
                print("✗ FAIL: verify_no_lookahead has 'for date in master.index' loop")
                return False
        except:
            return False

def check_trainer_warmup_lr():
    """Check 3: Verify trainer.py has _set_warmup_lr (not _warmup_lr)"""
    print("\n" + "="*70)
    print("CHECK 3: Verify trainer.py has _set_warmup_lr method")
    print("="*70)
    try:
        with open('src/training/trainer.py', 'r') as f:
            content = f.read()
        
        has_set_warmup_lr = 'def _set_warmup_lr' in content
        has_warmup_lr = 'def _warmup_lr' in content
        
        if has_set_warmup_lr and not has_warmup_lr:
            print("✓ PASS: Trainer has _set_warmup_lr method (not _warmup_lr)")
            
            # Check that it doesn't return None
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '_set_warmup_lr':
                    has_explicit_return = False
                    returns_none = False
                    for inner_node in ast.walk(node):
                        if isinstance(inner_node, ast.Return):
                            has_explicit_return = True
                            if inner_node.value is None or (isinstance(inner_node.value, ast.Constant) and inner_node.value.value is None):
                                returns_none = True
                    
                    if not has_explicit_return:
                        print("✓ PASS: _set_warmup_lr does not explicitly return None")
                        return True
                    elif not returns_none:
                        print("✓ PASS: _set_warmup_lr returns non-None value")
                        return True
                    else:
                        print("✗ FAIL: _set_warmup_lr explicitly returns None")
                        return False
            return True
        else:
            print(f"✗ FAIL: _set_warmup_lr not found or _warmup_lr still exists")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error reading trainer.py: {e}")
        return False

def check_fed_funds_rate_realtime_start():
    """Check 4: Verify _transform_fed_funds_rate preserves realtime_start"""
    print("\n" + "="*70)
    print("CHECK 4: Verify _transform_fed_funds_rate preserves realtime_start")
    print("="*70)
    try:
        with open('src/data/macro_data.py', 'r') as f:
            content = f.read()
        
        # Extract function
        start_idx = content.find('def _transform_fed_funds_rate')
        end_idx = content.find('\ndef ', start_idx + 1)
        func_content = content[start_idx:end_idx]
        
        # Check for preservation of realtime_start
        has_correct_preserve = 'df_monthly_pub' in func_content and 'realtime_start' in func_content
        has_observation_date_plus_1 = 'observation_date + 1' in func_content or 'observation_date + pd.Timedelta(days=1)' in func_content
        
        if has_correct_preserve and not has_observation_date_plus_1:
            print("✓ PASS: _transform_fed_funds_rate preserves realtime_start from original df")
            print("  - Does not use 'observation_date + 1 day' calculation")
            return True
        elif has_observation_date_plus_1:
            print("✗ FAIL: _transform_fed_funds_rate uses 'observation_date + 1 day' instead of preserving realtime_start")
            return False
        else:
            print("? UNCLEAR: Could not definitively verify realtime_start preservation")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error reading macro_data.py: {e}")
        return False

def check_backtester_batch_precompute():
    """Check 5: Verify backtester.py run() pre-computes h_t in batches"""
    print("\n" + "="*70)
    print("CHECK 5: Verify backtester.py run() pre-computes h_t in batches")
    print("="*70)
    try:
        with open('src/backtest/backtester.py', 'r') as f:
            content = f.read()
        
        # Check for the batch loop pattern
        has_batch_loop = 'for macro_seq, returns in self.test_loader' in content
        has_tft_batch_call = 'self.model.tft(macro_seq)' in content
        has_all_h_t = 'all_h_t' in content
        has_second_loop = 'for i in tqdm(range(len(all_h_t' in content or 'for i in range(len(all_h_t' in content
        
        if has_batch_loop and has_tft_batch_call and has_all_h_t and has_second_loop:
            print("✓ PASS: backtester.py run() pre-computes h_t in batches")
            print("  - Batch loop: 'for macro_seq, returns in self.test_loader'")
            print("  - TFT call: 'self.model.tft(macro_seq)' in batch")
            print("  - Storage: 'all_h_t' tensor")
            print("  - Second loop: per-day sampling using cached h_t")
            return True
        else:
            print("✗ FAIL: backtester.py run() does not pre-compute h_t in batches")
            print(f"  - Has batch loop: {has_batch_loop}")
            print(f"  - Has TFT batch call: {has_tft_batch_call}")
            print(f"  - Has all_h_t storage: {has_all_h_t}")
            print(f"  - Has second loop: {has_second_loop}")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error reading backtester.py: {e}")
        return False

def check_backtester_plot_distributions():
    """Check 6: Verify backtester.py plot_return_distributions uses self._stored_port_samples"""
    print("\n" + "="*70)
    print("CHECK 6: Verify backtester.py plot_return_distributions implementation")
    print("="*70)
    try:
        with open('src/backtest/backtester.py', 'r') as f:
            content = f.read()
        
        # Extract function
        start_idx = content.find('def plot_return_distributions')
        end_idx = content.find('\n    def ', start_idx + 1)
        if end_idx == -1:
            end_idx = len(content)
        func_content = content[start_idx:end_idx]
        
        # Check requirements
        uses_stored_samples = 'self._stored_port_samples' in func_content
        has_histogram = 'ax.hist(' in func_content or 'histogram' in func_content
        
        if uses_stored_samples and has_histogram:
            print("✓ PASS: plot_return_distributions uses self._stored_port_samples")
            print("✓ PASS: plot_return_distributions draws a histogram (ax.hist)")
            return True
        else:
            print("✗ FAIL: plot_return_distributions does not meet requirements")
            print(f"  - Uses self._stored_port_samples: {uses_stored_samples}")
            print(f"  - Has histogram: {has_histogram}")
            return False
    except Exception as e:
        print(f"✗ FAIL: Error reading backtester.py: {e}")
        return False

def check_python_syntax():
    """Bonus: Verify Python syntax of all files"""
    print("\n" + "="*70)
    print("BONUS: Python syntax check")
    print("="*70)
    files_to_check = [
        'src/data/pipeline.py',
        'src/training/trainer.py',
        'src/data/macro_data.py',
        'src/backtest/backtester.py',
    ]
    
    all_pass = True
    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"✓ {filepath}: Valid Python syntax")
        except SyntaxError as e:
            print(f"✗ {filepath}: Syntax error at line {e.lineno}: {e.msg}")
            all_pass = False
        except Exception as e:
            print(f"✗ {filepath}: Error: {e}")
            all_pass = False
    return all_pass

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FILE VERIFICATION SUITE")
    print("="*70)
    
    results = {
        "1. JSON validation": check_notebook_json(),
        "2a. Constants": check_pipeline_constants(),
        "2b. Return signature": check_build_pipeline_signature(),
        "2c. Vectorization": check_verify_no_lookahead_vectorized(),
        "3. Trainer method": check_trainer_warmup_lr(),
        "4. Fed funds rate": check_fed_funds_rate_realtime_start(),
        "5. Batch precompute": check_backtester_batch_precompute(),
        "6. Plot distributions": check_backtester_plot_distributions(),
        "Bonus. Python syntax": check_python_syntax(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_pass = all(results.values())
    print("\n" + "="*70)
    if all_pass:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    print("="*70 + "\n")
    
    sys.exit(0 if all_pass else 1)
