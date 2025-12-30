'use client';

import { RFQFileEntry } from '@/types/api';
import styles from './RFQFileList.module.css';

interface RFQFileListProps {
  files: RFQFileEntry[];
  onSelect: (file: RFQFileEntry) => void;
  onDelete: (id: string) => void;
}

function getFileIcon(sourceType: string): string {
  switch (sourceType) {
    case 'pdf':
      return '📕';
    case 'xlsx':
    case 'xls':
      return '📗';
    case 'csv':
      return '📘';
    default:
      return '📄';
  }
}

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export default function RFQFileList({ files, onSelect, onDelete }: RFQFileListProps) {
  return (
    <div className={styles.container}>
      <h2 className={styles.heading}>Uploaded Files</h2>
      <div className={styles.table}>
        <div className={styles.headerRow}>
          <div className={styles.headerCell}>File</div>
          <div className={styles.headerCell}>Type</div>
          <div className={styles.headerCell}>Items</div>
          <div className={styles.headerCell}>Uploaded</div>
          <div className={styles.headerCell}></div>
        </div>
        {files.map((file) => (
          <div
            key={file.id}
            className={styles.row}
            onClick={() => onSelect(file)}
          >
            <div className={styles.cell}>
              <span className={styles.icon}>{getFileIcon(file.sourceType)}</span>
              <span className={styles.filename}>{file.filename}</span>
            </div>
            <div className={styles.cell}>
              <span className={styles.badge}>{file.sourceType.toUpperCase()}</span>
            </div>
            <div className={styles.cell}>
              <span className={styles.count}>{file.totalItems}</span>
            </div>
            <div className={styles.cell}>
              <span className={styles.date}>{formatDate(file.uploadedAt)}</span>
            </div>
            <div className={styles.cell}>
              <button
                className={styles.deleteBtn}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(file.id);
                }}
                title="Remove"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

