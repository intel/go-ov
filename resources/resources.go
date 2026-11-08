// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package resources

import (
	"embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

//go:embed *
var Content embed.FS

// TODO: when building ov_worker compute its checksum and save it to a file on the resources.
// then when extracting the worker, verify its checksum to ensure integrity. Also when
// checking if the worker already exists, verify its checksum as well.

// Extract writes the named embedded file into dir. If the file already exists
// in dir it is not written again. It returns the full path to the extracted file.
func Extract(filename, dir string, overwrite bool) (string, error) {
	path := filepath.Join(dir, filename)

	proceed, err := ensureWritableTarget(path, overwrite)
	if err != nil {
		return "", err
	}
	if !proceed {
		return path, nil
	}

	data, err := Content.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("read embedded file %q: %w", filename, err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create directory %q: %w", dir, err)
	}
	if err := os.WriteFile(path, data, 0o755); err != nil {
		return "", fmt.Errorf("write file %q: %w", path, err)
	}
	return path, nil
}

// CopyDir recursively copies the contents of srcDir into dstDir, streaming each file so large
// binaries are not buffered in memory. Existing destination files are overwritten only when
// overwrite is true; otherwise they are left untouched.
func CopyDir(srcDir, dstDir string, overwrite bool) error {
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return fmt.Errorf("read source directory %q: %w", srcDir, err)
	}
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return fmt.Errorf("create directory %q: %w", dstDir, err)
	}
	for _, entry := range entries {
		srcPath := filepath.Join(srcDir, entry.Name())
		dstPath := filepath.Join(dstDir, entry.Name())
		if entry.IsDir() {
			if err := CopyDir(srcPath, dstPath, overwrite); err != nil {
				return err
			}
			continue
		}
		proceed, err := ensureWritableTarget(dstPath, overwrite)
		if err != nil {
			return err
		}
		if !proceed {
			continue // existing regular file kept (not overwriting)
		}
		if err := copyFile(srcPath, dstPath); err != nil {
			return err
		}
	}
	return nil
}

// copyFile streams the contents of srcPath into dstPath.
func copyFile(srcPath, dstPath string) error {
	in, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("open %q: %w", srcPath, err)
	}
	defer in.Close()

	out, err := os.Create(dstPath)
	if err != nil {
		return fmt.Errorf("create %q: %w", dstPath, err)
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return fmt.Errorf("copy to %q: %w", dstPath, err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("close %q: %w", dstPath, err)
	}
	return nil
}

// ensureWritableTarget prepares path for a fresh write without ever following a symlink. An
// existing symlink is unexpected here, so it is removed and the real file replaces it (even when
// overwrite is false). It returns true if the caller should write; false means an existing
// regular file must be kept. Non-regular files (e.g. directories) are rejected.
func ensureWritableTarget(path string, overwrite bool) (bool, error) {
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return true, nil
		}
		return false, fmt.Errorf("inspect %q: %w", path, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		if err := os.Remove(path); err != nil {
			return false, fmt.Errorf("remove symlink %q: %w", path, err)
		}
		return true, nil
	}
	if !info.Mode().IsRegular() {
		return false, fmt.Errorf("refusing to overwrite non-regular file %q", path)
	}
	if !overwrite {
		return false, nil
	}
	return true, nil
}
